import pygame
import math
import random
import pickle
import os
import sys
from utils import scale_image, blit_rotate_center

################## Pygame Setup ##################
pygame.font.init()
STAT_FONT = pygame.font.SysFont("consolas", 18)
TITLE_FONT = pygame.font.SysFont("consolas", 24, bold=True)

try:
    GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
    TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
    TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
    TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
    FINISH = pygame.image.load("imgs/finish.png")
    RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
except Exception as e:
    print(f"Error loading images. {e}")
    sys.exit()

FINISH_POSITION = (130, 250)
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Trainer: Concurrent Optimization")

FPS = 60
N_CARS = 20

#### basics ########

ACTIONS = ["ACCELERATE", "BRAKE", "ROTATE_LEFT", "ROTATE_RIGHT", "GO_STRAIGHT"]
ACTION_REPEAT = 4
CHECKPOINT_RADIUS = 5

CHECKPOINTS = [
    (176, 129),
    (121, 81),
    (62, 125),
    (61, 444),
    (249, 674),
    (404, 631),
    (502, 487),
    (600, 613),
    (739, 642),
    (737, 443),
    (647, 365),
    (475, 367),
    (407, 315),
    (480, 265),
    (641, 257),
    (653, 78),
    (345, 76),
    (281, 158),
    (280, 329),
    (176, 345)
]

##### Car class ####

class Car:
    IMG = RED_CAR
    START_POS = (180, 200)

    def __init__(self, max_vel, rotation_vel):
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.img = self.IMG
        self.reset()

    def reset(self):
        self.vel = 0
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.alive = True

    def get_state(self):
        return {
            "x": self.x, "y": self.y,
            "angle": self.angle, "vel": self.vel,
            "alive": self.alive
        }

    def set_state(self, state):
        self.x = state["x"]
        self.y = state["y"]
        self.angle = state["angle"]
        self.vel = state["vel"]
        self.alive = state["alive"]

    def rotate(self, left=False, right=False):
        if left: self.angle += self.rotation_vel
        elif right: self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def reduce_speed(self):
        if self.vel > 0:
            self.vel = max(self.vel - self.acceleration/2, 0)
        else:
            self.vel = min(self.vel + self.acceleration/2, 0)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizontal
        self.check_collision()

    def check_collision(self):
        if self.collide(TRACK_BORDER_MASK) is not None:
            self.alive = False

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

class PlayerCar(Car):
    def update_manual(self):
        keys = pygame.key.get_pressed()
        moved = False
        if keys[pygame.K_a]: self.rotate(left=True)
        if keys[pygame.K_d]: self.rotate(right=True)
        if keys[pygame.K_w]: moved = True; self.move_forward()
        if keys[pygame.K_s]: moved = True; self.move_backward()
        if not moved: self.reduce_speed()

##### AI MANAGER ######

class TrainingManager:
    def __init__(self):
        self.committed_actions = []
        self.committed_path_points = [(180, 200)]
        self.start_state = None
        self.start_checkpoint_idx = 0

        self.history_stack = []

        self.current_segment_actions = self.random_actions(150)
        self.best_segment_actions = list(self.current_segment_actions)
        self.best_segment_score = -99999
        self.accepted_score = -99999

        self.stagnation_counter = 0
        self.STAGNATION_THRESHOLD = 20
        self.optimizing_full_lap = False
        self.best_lap_time = 99999

        self.sim_states = []
        self.step_index_global = 0

        self.INITIAL_TEMP = 200
        self.T = self.INITIAL_TEMP
        self.cooling = 0.99
        self.mode = "COOLING"

        self.load_model()

    def random_actions(self, length):
        return [random.choice(ACTIONS) for _ in range(length)]

    def full_reset(self):
        if os.path.exists("saved_state.pkl"):
            os.remove("saved_state.pkl")
        self.__init__()
        print("TRAINING RESET.")

    def undo_last_segment(self, cars):
        if self.optimizing_full_lap:
            print("Cannot undo during full optimization.")
            return
        if not self.history_stack:
            print("Nothing to undo!")
            return
        print("Undoing last segment...")
        last_state, actions_len, path_len = self.history_stack.pop()
        self.start_state = last_state
        self.committed_actions = self.committed_actions[:actions_len]
        self.committed_path_points = self.committed_path_points[:path_len]
        self.start_checkpoint_idx = (self.start_checkpoint_idx - 1) % len(CHECKPOINTS)
        self.reset_car_to_segment_start(cars)
        self.save_model()

    def start_full_optimization(self, cars):
        print(f"ENTERING OPTIMIZE MODE. Current Path Length: {len(self.committed_actions)} frames")
        self.optimizing_full_lap = True
        self.current_segment_actions = list(self.committed_actions)
        self.current_segment_actions.extend(self.random_actions(100))
        self.best_segment_actions = list(self.current_segment_actions)
        self.start_state = None
        self.start_checkpoint_idx = 0
        self.T = 50
        self.mode = "PRECISION"
        self.best_segment_score = -99999
        self.accepted_score = -99999
        self.reset_car_to_segment_start(cars)

    def load_model(self):
        if os.path.exists("saved_state.pkl"):
            try:
                with open("saved_state.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.committed_actions = data["committed_actions"]
                    self.committed_path_points = data.get("committed_path", [(180, 200)])
                    self.start_state = data["start_state"]
                    self.start_checkpoint_idx = data["start_checkpoint_idx"]
                    self.history_stack = data.get("history_stack", [])
                    self.best_lap_time = data.get("best_lap_time", 99999)
                    print("Hot Resume: Loaded previous state.")
            except:
                print("Could not load save file.")

    def save_model(self):
        data = {
            "committed_actions": self.committed_actions,
            "committed_path": self.committed_path_points,
            "start_state": self.start_state,
            "start_checkpoint_idx": self.start_checkpoint_idx,
            "history_stack": self.history_stack,
            "best_lap_time": self.best_lap_time
        }
        with open("saved_state.pkl", "wb") as f:
            pickle.dump(data, f)

    def create_mutated_actions(self):
        actions = list(self.current_segment_actions)
        if self.mode == "PRECISION": rate = 1
        elif self.mode == "PANIC": rate = 5
        else: rate = 1 if self.T <= 100 else 3

        for _ in range(random.randint(1, rate)):
            idx = random.randint(0, len(actions)-1)
            actions[idx] = random.choice(ACTIONS)

        if self.mode != "PRECISION" and random.random() < 0.2:
            block_size = random.randint(5, 10)
            if len(actions) > block_size:
                start_idx = random.randint(0, len(actions) - block_size)
                forced_action = random.choice(["ROTATE_LEFT", "ROTATE_RIGHT", "ACCELERATE"])
                for i in range(start_idx, start_idx + block_size):
                    actions[i] = forced_action
        return actions

    def reset_car_to_segment_start(self, cars):
        self.sim_states = []
        target = CHECKPOINTS[self.start_checkpoint_idx]

        for i, car in enumerate(cars):
            if self.start_state:
                car.set_state(self.start_state)
            else:
                car.reset()

            if i == 0 and self.mode != "PANIC":
                 mutated_actions = list(self.current_segment_actions)
            else:
                 mutated_actions = self.create_mutated_actions()

            prev_dist = math.hypot(car.x - target[0], car.y - target[1])

            state = {
                "step_index": 0,
                "frame_counter": 0,
                "current_reward": 0,
                "current_path_points": [],
                "prev_distance": prev_dist,
                "closest_dist_this_run": prev_dist,
                "actions": mutated_actions,
                "alive": True,
                "finished": False,
                "car_ref": car
            }
            self.sim_states.append(state)

    def get_smart_initialization(self, car):
        target = CHECKPOINTS[(self.start_checkpoint_idx) % len(CHECKPOINTS)]
        dx = target[0] - car.x
        dy = target[1] - car.y
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        diff = (target_angle - car.angle + 180) % 360 - 180

        actions = []
        for _ in range(150):
            r = random.random()
            if r < 0.6:
                if diff > 10: actions.append("ROTATE_LEFT")
                elif diff < -10: actions.append("ROTATE_RIGHT")
                else: actions.append("ACCELERATE")
            else:
                actions.append(random.choice(ACTIONS))
        return actions

    def calculate_entry_angle_score(self, car):
        next_idx = (self.start_checkpoint_idx + 1) % len(CHECKPOINTS)
        next_target = CHECKPOINTS[next_idx]
        dx = next_target[0] - car.x
        dy = next_target[1] - car.y
        ideal_angle = math.degrees(math.atan2(dy, dx)) + 90
        diff = abs((ideal_angle - car.angle + 180) % 360 - 180)
        return max(0, 1.0 - (diff / 180.0))

    def commit_segment(self, state, cars):
        car = state["car_ref"]
        if self.start_checkpoint_idx == len(CHECKPOINTS) - 1:
            print("LAP FINISHED! Saving full run.")
            actions_taken = state["actions"][:state["step_index"]+1]
            self.committed_actions.extend(actions_taken)
            self.committed_path_points.extend(state["current_path_points"])

            total_frames = len(self.committed_actions) * ACTION_REPEAT
            self.best_lap_time = total_frames / 60
            self.start_full_optimization(cars)
            return

        self.history_stack.append((
            self.start_state,
            len(self.committed_actions),
            len(self.committed_path_points)
        ))

        actions_taken = state["actions"][:state["step_index"]+1]
        self.committed_actions.extend(actions_taken)
        self.committed_path_points.extend(state["current_path_points"])

        self.start_state = car.get_state()
        self.start_checkpoint_idx = (self.start_checkpoint_idx + 1) % len(CHECKPOINTS)

        print(f"Checkpoint {self.start_checkpoint_idx} Reached! Committed.")

        self.current_segment_actions = self.get_smart_initialization(car)
        self.best_segment_actions = list(self.current_segment_actions)
        self.best_segment_score = -99999
        self.accepted_score = -99999
        self.stagnation_counter = 0
        self.T = self.INITIAL_TEMP
        self.mode = "COOLING"

        self.save_model()
        self.reset_car_to_segment_start(cars)

    def prepare_next_attempt(self, cars):
        finished_states = [s for s in self.sim_states if s["finished"]]

        if finished_states:
            best_finisher = max(finished_states, key=lambda s: s["current_reward"])
            self.commit_segment(best_finisher, cars)
            return

        best_sim_state = max(self.sim_states, key=lambda s: s["current_reward"])

        score = best_sim_state["current_reward"]
        used_actions = best_sim_state["actions"]
        self.step_index_global = best_sim_state["step_index"]

        if score > self.best_segment_score:
            self.best_segment_score = score
            self.best_segment_actions = list(used_actions)
            self.stagnation_counter = 0
            if self.optimizing_full_lap:
                total_frames = best_sim_state["step_index"] * ACTION_REPEAT
                self.best_lap_time = total_frames / 60
        else:
            self.stagnation_counter += 1

        if best_sim_state["closest_dist_this_run"] < 80 or self.stagnation_counter > 20:
             self.mode = "PRECISION"; self.T = 20
        elif self.stagnation_counter > 50:
             self.mode = "PANIC"; self.T = 400
        else:
             self.mode = "COOLING"; self.T = max(10, self.T * self.cooling)

        if self.optimizing_full_lap: self.mode = "PRECISION"; self.T = 20

        if self.mode == "PRECISION":
            if score >= self.best_segment_score:
                self.accepted_score = score; self.current_segment_actions = list(used_actions)
            else:
                self.current_segment_actions = list(self.best_segment_actions)
        else:
            if score > self.accepted_score or self.accepted_score == -99999:
                self.accepted_score = score
                self.best_segment_actions = list(used_actions)
                self.current_segment_actions = list(used_actions)
            else:
                try: prob = math.exp((score - self.accepted_score) / self.T)
                except OverflowError: prob = 0
                if random.random() < prob:
                    self.accepted_score = score; self.current_segment_actions = list(used_actions)
                else:
                    self.current_segment_actions = list(self.best_segment_actions)

        if not self.optimizing_full_lap and self.stagnation_counter > 60:
             self.current_segment_actions = self.get_smart_initialization(cars[0])
             self.stagnation_counter = 0

        self.reset_car_to_segment_start(cars)

    def update(self, cars):
        active_cars = 0

        for i, car in enumerate(cars):
            state = self.sim_states[i]

            if not state["alive"] or state["finished"]:
                continue

            active_cars += 1

            if state["step_index"] >= len(state["actions"]):
                 if self.optimizing_full_lap:
                     state["current_reward"] += 5000
                     state["finished"] = True
                 else:
                     state["current_reward"] -= 200
                     state["alive"] = False
                 continue

            # Physics stuff
            action = state["actions"][state["step_index"]]
            match action:
                case "ACCELERATE": car.move_forward()
                case "BRAKE": car.move_backward()
                case "GO_STRAIGHT":
                    if car.vel > 0: car.reduce_speed()
                    else: car.vel = 0; car.move()
                case "ROTATE_LEFT": car.rotate(left=True); car.move()
                case "ROTATE_RIGHT": car.rotate(right=True); car.move()

            # Path Vis
            if state["frame_counter"] == 0:
                state["current_path_points"].append((car.x + car.img.get_width()/2, car.y + car.img.get_height()/2))

            # Reward calculations
            target = CHECKPOINTS[self.start_checkpoint_idx]
            dist = math.hypot(car.x - target[0], car.y - target[1])
            if dist < state["closest_dist_this_run"]: state["closest_dist_this_run"] = dist

            improvement = state["prev_distance"] - dist
            state["current_reward"] += (improvement * 5)
            state["prev_distance"] = dist

            if car.vel < 0 and improvement <= 0: state["current_reward"] -= 0.1

            if car.collide(TRACK_BORDER_MASK) is not None:
                car.alive = False; state["alive"] = False
                state["current_reward"] -= 1000
                continue

            dist_center = math.hypot(car.x - target[0], car.y - target[1])

            if dist_center < 60:
                car_mask = pygame.mask.from_surface(car.img)
                cp_surface = pygame.Surface((CHECKPOINT_RADIUS*2, CHECKPOINT_RADIUS*2), pygame.SRCALPHA)
                pygame.draw.circle(cp_surface, (255, 255, 255), (CHECKPOINT_RADIUS, CHECKPOINT_RADIUS), CHECKPOINT_RADIUS)
                cp_mask = pygame.mask.from_surface(cp_surface)
                offset = (int((target[0] - CHECKPOINT_RADIUS) - car.x), int((target[1] - CHECKPOINT_RADIUS) - car.y))

                if car_mask.overlap(cp_mask, offset):
                    if self.optimizing_full_lap:
                        state["current_reward"] += 2000
                        self.start_checkpoint_idx = (self.start_checkpoint_idx + 1) % len(CHECKPOINTS)
                    else:
                        angle_quality = self.calculate_entry_angle_score(car)
                        state["current_reward"] += 2000 + (1000 * angle_quality)
                        if angle_quality < 0.3: state["current_reward"] -= 500

                    state["finished"] = True
                    continue

            state["frame_counter"] += 1
            if state["frame_counter"] >= ACTION_REPEAT:
                state["step_index"] += 1
                state["frame_counter"] = 0

        if active_cars == 0:
            self.prepare_next_attempt(cars)

#### VISUALIZATION & MAIN #####

def draw_paths(win, manager):
    if not manager.sim_states: return

    best_state = max(manager.sim_states, key=lambda s: s["current_reward"])

    if manager.optimizing_full_lap and len(best_state["current_path_points"]) > 1:
         pygame.draw.lines(win, (0, 255, 0), False, best_state["current_path_points"], 3)
    else:
        if len(manager.committed_path_points) > 1:
            pygame.draw.lines(win, (0, 255, 0), False, manager.committed_path_points, 3)
        if len(best_state["current_path_points"]) > 1:
            pygame.draw.lines(win, (255, 50, 50), False, best_state["current_path_points"], 2)

def draw_ui(win, mode, manager, hyperspeed):
    bg_rect = pygame.Rect(0, HEIGHT - 180, 320, 180)
    pygame.draw.rect(win, (0, 0, 0), bg_rect)
    pygame.draw.rect(win, (255, 255, 255), bg_rect, 2)

    speed_txt = "HYPERSPEED" if hyperspeed else "Normal Speed"
    col = (255, 100, 100) if hyperspeed else (200, 200, 200)

    status_col = (255, 255, 255)
    if manager.mode == "PRECISION": status_col = (0, 255, 0)
    elif manager.mode == "PANIC": status_col = (255, 0, 0)

    lap_time_str = f"{manager.best_lap_time:.2f}s" if manager.best_lap_time < 9999 else "--"
    current_reward_disp = 0
    if manager.sim_states:
        current_reward_disp = max(s["current_reward"] for s in manager.sim_states)

    lines = [
        f"Mode: {mode} | {speed_txt} | x{N_CARS}",
        f"Lap Time: {lap_time_str}",
        f"CP: {manager.start_checkpoint_idx} | {manager.mode}",
        f"Stagnation: {manager.stagnation_counter}",
        f"Reward: {int(current_reward_disp)}",
        "KEYS: [T]rain, [M]anual, [H]yperspeed",
        "      [O]ptimize Full, [BKSPC] Reset",
        "      [U] Undo Last Checkpoint"
    ]

    for i, line in enumerate(lines):
        c = col if i == 0 else (255, 255, 255)
        if "CP:" in line: c = status_col
        if "Lap Time" in line: c = (255, 215, 0)
        txt = STAT_FONT.render(line, 1, c)
        win.blit(txt, (10, HEIGHT - 170 + (i * 20)))

    if manager.optimizing_full_lap:
        lbl = TITLE_FONT.render("OPTIMIZING LAP...", 1, (0, 255, 0))
        win.blit(lbl, (WIDTH/2 - lbl.get_width()/2, 50))

def draw(win, images, cars, mode, manager, hyperspeed):
    for img, pos in images:
        win.blit(img, pos)

    pygame.draw.line(win, (255, 255, 255), (130, 250), (130, 370), 5)

    if mode == "TRAINING":
        draw_paths(win, manager)
        target = CHECKPOINTS[manager.start_checkpoint_idx]
        pygame.draw.circle(win, (0, 255, 255), target, CHECKPOINT_RADIUS, 2)

    if isinstance(cars, list) and manager.sim_states:
        indexed_rewards = []
        for i, s in enumerate(manager.sim_states):
            indexed_rewards.append((i, s["current_reward"]))
        indexed_rewards.sort(key=lambda x: x[1])

        for i, _ in indexed_rewards:
            c = cars[i]
            if c.alive or manager.sim_states[i]["finished"]:
                c.draw(win)
    else:
        if cars.alive: cars.draw(win)

    draw_ui(win, mode, manager, hyperspeed)
    pygame.display.update()

def main():
    run = True
    clock = pygame.time.Clock()

    images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

    player_car = PlayerCar(4, 4)
    ai_cars = [Car(4, 4) for _ in range(N_CARS)]
    manager = TrainingManager()

    mode = "MANUAL"
    hyperspeed = False

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    mode = "MANUAL"
                    player_car.reset()
                if event.key == pygame.K_t:
                    mode = "TRAINING"
                    manager.reset_car_to_segment_start(ai_cars)
                if event.key == pygame.K_h:
                    hyperspeed = not hyperspeed

                if event.key == pygame.K_BACKSPACE:
                    manager.full_reset()
                    for car in ai_cars: car.reset()
                if event.key == pygame.K_o:
                    mode = "TRAINING"
                    manager.start_full_optimization(ai_cars)
                if event.key == pygame.K_u:
                    mode = "TRAINING"
                    manager.undo_last_segment(ai_cars)

        loops = 100 if (hyperspeed and mode == "TRAINING") else 1

        for _ in range(loops):
            if mode == "MANUAL":
                player_car.update_manual()
                player_car.check_collision()
                if not player_car.alive: player_car.reset()

            elif mode == "TRAINING":
                manager.update(ai_cars)

        cars_to_draw = player_car if mode == "MANUAL" else ai_cars
        draw(WIN, images, cars_to_draw, mode, manager, hyperspeed)

    manager.save_model()
    pygame.quit()

if __name__ == "__main__":
    main()
