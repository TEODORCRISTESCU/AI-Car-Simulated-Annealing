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
pygame.display.set_caption("AI Trainer: Optimize Fix & Lap Time")

FPS = 60

#### basics ########

ACTIONS = ["ACCELERATE", "BRAKE", "ROTATE_LEFT", "ROTATE_RIGHT", "GO_STRAIGHT"]
ACTION_REPEAT = 4
CHECKPOINT_RADIUS = 15 # so that we can make the moves more accurate

CHECKPOINTS = [
    (116, 71), (49, 138), (70, 481), (315, 732),
    (408, 671), (429, 499), (571, 504), (611, 705),
    (727, 713), (727, 386), (438, 360), (410, 287),
    (721, 242), (726, 90), (298, 81), (280, 362),
    (209, 401), (178, 359)
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
        if self.collide(TRACK_BORDER_MASK) is not None: # see if car hit a wall
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
        # Saved state
        self.committed_actions = []
        self.committed_path_points = [(180, 200)]
        self.start_state = None
        self.start_checkpoint_idx = 0

        # Undo history
        self.history_stack = []

        # Current Segment Training
        self.current_segment_actions = self.random_actions(150)
        self.best_segment_actions = list(self.current_segment_actions)
        self.best_segment_score = -99999
        self.accepted_score = -99999

        # Logic Flags
        self.stagnation_counter = 0
        self.STAGNATION_THRESHOLD = 20
        self.optimizing_full_lap = False
        self.best_lap_time = 99999

        # Simulation
        self.step_index = 0
        self.frame_counter = 0
        self.current_reward = 0
        self.current_path_points = []
        self.prev_distance = 0
        self.closest_dist_this_run = 9999

        # Annealing
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

    def undo_last_segment(self, car):
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
        self.reset_car_to_segment_start(car)
        self.save_model()

    def start_full_optimization(self, car):
        print(f"ENTERING OPTIMIZE MODE. Current Path Length: {len(self.committed_actions)} frames")

        self.optimizing_full_lap = True

        # Load the full path as the "Current Segment"
        self.current_segment_actions = list(self.committed_actions)

        # Ensure it has enough buffer to finish if it speeds up
        self.current_segment_actions.extend(self.random_actions(100))

        self.best_segment_actions = list(self.current_segment_actions)

        # Reset state to beginning of track
        self.start_state = None
        self.start_checkpoint_idx = 0

        # Use Low Temp for fine tuning (we don't want to break the path, just polish it)
        self.T = 50
        self.mode = "PRECISION"

        self.best_segment_score = -99999
        self.accepted_score = -99999
        self.reset_car_to_segment_start(car)

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
                print("Could not load save file => Start again, please!.")

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

    def reset_car_to_segment_start(self, car):
        if self.start_state:
            car.set_state(self.start_state)
        else:
            car.reset()

        target = CHECKPOINTS[self.start_checkpoint_idx]
        self.prev_distance = math.hypot(car.x - target[0], car.y - target[1])
        self.closest_dist_this_run = self.prev_distance

        self.step_index = 0
        self.frame_counter = 0
        self.current_reward = 0
        self.current_path_points = []

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

    def commit_segment(self, car):
        # We finished the lap!
        if self.start_checkpoint_idx == len(CHECKPOINTS) - 1:
            print("LAP FINISHED! Saving full run. Switching to Optimization.")

            # Commit the final piece
            actions_taken = self.current_segment_actions[:self.step_index+1]
            self.committed_actions.extend(actions_taken)
            self.committed_path_points.extend(self.current_path_points)

            # Calculate Lap Time (Frames * Repeat / 60)
            total_frames = len(self.committed_actions) * ACTION_REPEAT
            seconds = total_frames / 60
            self.best_lap_time = seconds

            # Trigger Auto-Optimize
            self.start_full_optimization(car)
            return

        # Normal Segment Commit
        self.history_stack.append((
            self.start_state,
            len(self.committed_actions),
            len(self.committed_path_points)
        ))

        actions_taken = self.current_segment_actions[:self.step_index+1]
        self.committed_actions.extend(actions_taken)
        self.committed_path_points.extend(self.current_path_points)

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

    def prepare_next_attempt(self, car):
        score = self.current_reward

        ### Updates best attempt ####
        if score > self.best_segment_score:
            self.best_segment_score = score
            self.best_segment_actions = list(self.current_segment_actions)
            self.stagnation_counter = 0

            # If optimizing, update committed actions immediately to show improvement
            if self.optimizing_full_lap:
                # Update lap time display
                total_frames = self.step_index * ACTION_REPEAT
                self.best_lap_time = total_frames / 60
                print(f"Optimization Improvement! New Time: {self.best_lap_time:.2f}s")
        else:
            self.stagnation_counter += 1

        # ### Mode Selection ####
        if self.closest_dist_this_run < 80 or self.stagnation_counter > 20:
             self.mode = "PRECISION"
             self.T = 20
        elif self.stagnation_counter > 50:
             self.mode = "PANIC"
             self.T = 400
        else:
             self.mode = "COOLING"
             self.T = max(10, self.T * self.cooling)

        # Force Precision during full optimization (don't break the lap)
        if self.optimizing_full_lap:
            self.mode = "PRECISION"
            self.T = 20

        #### ACCEPTANCE ####
        if self.mode == "PRECISION":
            if score >= self.best_segment_score:
                self.accepted_score = score
                self.current_segment_actions = list(self.current_segment_actions)
            else:
                self.current_segment_actions = list(self.best_segment_actions)
        else:
            if score > self.accepted_score or self.accepted_score == -99999:
                self.accepted_score = score
                self.best_segment_actions = list(self.current_segment_actions)
            else:
                try:
                    prob = math.exp((score - self.accepted_score) / self.T)
                except OverflowError:
                    prob = 0

                if random.random() < prob:
                    self.accepted_score = score
                else:
                    self.current_segment_actions = list(self.best_segment_actions)

        # Panic Scramble
        if not self.optimizing_full_lap and self.stagnation_counter > 60:
             self.current_segment_actions = self.get_smart_initialization(car)
             self.stagnation_counter = 0

        self.mutate_actions()
        self.reset_car_to_segment_start(car)

    def mutate_actions(self):
        if self.mode == "PRECISION":
            rate = 1
        elif self.mode == "PANIC":
            rate = 5
        else:
            rate = 1
            if self.T > 100: rate = 3

        for _ in range(random.randint(1, rate)):
            idx = random.randint(0, len(self.current_segment_actions)-1)
            self.current_segment_actions[idx] = random.choice(ACTIONS)

        if self.mode != "PRECISION" and random.random() < 0.2:
            block_size = random.randint(5, 10)
            start_idx = random.randint(0, len(self.current_segment_actions) - block_size)
            forced_action = random.choice(["ROTATE_LEFT", "ROTATE_RIGHT", "ACCELERATE"])
            for i in range(start_idx, start_idx + block_size):
                self.current_segment_actions[i] = forced_action

    def update(self, car):

        if not car.alive:
            self.current_reward -= 1000
            self.prepare_next_attempt(car)
            return

        if self.step_index >= len(self.current_segment_actions):
            # If optimizing, running out of actions means we finished the lap (Success!)
            if self.optimizing_full_lap:
                self.current_reward += 5000 # Big bonus for finishing
                self.prepare_next_attempt(car) # Try to do it faster next time
            else:
                self.current_reward -= 200
                self.prepare_next_attempt(car)
            return

        action = self.current_segment_actions[self.step_index]
        match action:
            case "ACCELERATE": car.move_forward()
            case "BRAKE": car.move_backward()
            case "GO_STRAIGHT":
                if car.vel > 0: car.reduce_speed()
                else: car.vel = 0; car.move()
            case "ROTATE_LEFT": car.rotate(left=True); car.move()
            case "ROTATE_RIGHT": car.rotate(right=True); car.move()

        # 3. Visualization Path
        if self.frame_counter == 0:
            self.current_path_points.append((car.x + car.img.get_width()/2, car.y + car.img.get_height()/2))

        # 4. Target & Reward Logic
        current_cp_index = self.start_checkpoint_idx
        if self.optimizing_full_lap:
            # Dynamically find closest checkpoint to allow lap tracking
            pass

        target = CHECKPOINTS[current_cp_index]
        dist = math.hypot(car.x - target[0], car.y - target[1])

        if dist < self.closest_dist_this_run:
            self.closest_dist_this_run = dist

        improvement = self.prev_distance - dist
        self.current_reward += (improvement * 5)
        self.prev_distance = dist

        if car.vel < 0:
            if improvement <= 0:
                self.current_reward -= 0.1

        dist_center = math.hypot(car.x - target[0], car.y - target[1])

        # 5. Checkpoint Hit Logic
        hit_checkpoint = False
        if dist_center < 60:
            car_mask = pygame.mask.from_surface(car.img)

            # Create a surface for the checkpoint to make a mask
            cp_surface = pygame.Surface((CHECKPOINT_RADIUS*2, CHECKPOINT_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(cp_surface, (255, 255, 255), (CHECKPOINT_RADIUS, CHECKPOINT_RADIUS), CHECKPOINT_RADIUS)
            cp_mask = pygame.mask.from_surface(cp_surface)

            # Calculate offset (Checkpoint TopLeft - Car TopLeft)
            offset_x = (target[0] - CHECKPOINT_RADIUS) - car.x
            offset_y = (target[1] - CHECKPOINT_RADIUS) - car.y

            # Check overlap
            if car_mask.overlap(cp_mask, (int(offset_x), int(offset_y))):
                hit_checkpoint = True

        if hit_checkpoint:
            if self.optimizing_full_lap:
                self.current_reward += 2000
                self.start_checkpoint_idx = (self.start_checkpoint_idx + 1) % len(CHECKPOINTS)
            else:
                angle_quality = self.calculate_entry_angle_score(car)
                self.current_reward += 2000 + (1000 * angle_quality)
                if angle_quality < 0.3: self.current_reward -= 500

                self.commit_segment(car)
                self.reset_car_to_segment_start(car)
                return

        self.frame_counter += 1
        if self.frame_counter >= ACTION_REPEAT:
            self.step_index += 1
            self.frame_counter = 0

#### VISUALIZATION & MAIN #####

def draw_paths(win, manager):
    # If optimizing, we draw the best lap attempt
    if manager.optimizing_full_lap and len(manager.current_path_points) > 1:
         pygame.draw.lines(win, (0, 255, 0), False, manager.current_path_points, 3)
    else:
        if len(manager.committed_path_points) > 1:
            pygame.draw.lines(win, (0, 255, 0), False, manager.committed_path_points, 3)
        if len(manager.current_path_points) > 1:
            pygame.draw.lines(win, (255, 50, 50), False, manager.current_path_points, 2)

def draw_ui(win, mode, manager, hyperspeed):
    bg_rect = pygame.Rect(0, HEIGHT - 180, 320, 180)
    pygame.draw.rect(win, (0, 0, 0), bg_rect)
    pygame.draw.rect(win, (255, 255, 255), bg_rect, 2)

    speed_txt = "HYPERSPEED" if hyperspeed else "Normal Speed"
    col = (255, 100, 100) if hyperspeed else (200, 200, 200)

    # Color code the status
    status_col = (255, 255, 255)
    if manager.mode == "PRECISION": status_col = (0, 255, 0)
    elif manager.mode == "PANIC": status_col = (255, 0, 0)

    lap_time_str = f"{manager.best_lap_time:.2f}s" if manager.best_lap_time < 9999 else "--"

    lines = [
        f"Mode: {mode} | {speed_txt}",
        f"Lap Time: {lap_time_str}",
        f"CP: {manager.start_checkpoint_idx} | {manager.mode}",
        f"Stagnation: {manager.stagnation_counter}",
        f"Reward: {int(manager.current_reward)}",
        "KEYS: [T]rain, [M]anual, [H]yperspeed",
        "      [O]ptimize Full, [BKSPC] Reset",
        "      [U] Undo Last Checkpoint"
    ]

    for i, line in enumerate(lines):
        c = col if i == 0 else (255, 255, 255)
        if "CP:" in line: c = status_col
        if "Lap Time" in line: c = (255, 215, 0) # Gold color
        txt = STAT_FONT.render(line, 1, c)
        win.blit(txt, (10, HEIGHT - 170 + (i * 20)))

    if manager.optimizing_full_lap:
        lbl = TITLE_FONT.render("OPTIMIZING LAP...", 1, (0, 255, 0))
        win.blit(lbl, (WIDTH/2 - lbl.get_width()/2, 50))

def draw(win, images, car, mode, manager, hyperspeed):
    for img, pos in images:
        win.blit(img, pos)

    # Draw Finish Line
    pygame.draw.line(win, (255, 255, 255), (130, 250), (130, 370), 5)

    if mode == "TRAINING":
        draw_paths(win, manager)   # Draw target circles
        target = CHECKPOINTS[manager.start_checkpoint_idx]
        pygame.draw.circle(win, (0, 255, 255), target, CHECKPOINT_RADIUS, 2)

    if car.alive:
        car.draw(win)

    draw_ui(win, mode, manager, hyperspeed)
    pygame.display.update()

def main():
    run = True
    clock = pygame.time.Clock()

    images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

    player_car = PlayerCar(4, 4)
    ai_car = Car(4, 4)
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
                    manager.reset_car_to_segment_start(ai_car)
                if event.key == pygame.K_h:
                    hyperspeed = not hyperspeed

                if event.key == pygame.K_BACKSPACE:
                    manager.full_reset()
                    ai_car.reset()
                if event.key == pygame.K_o:
                    mode = "TRAINING"
                    manager.start_full_optimization(ai_car)
                if event.key == pygame.K_u:
                    mode = "TRAINING"
                    manager.undo_last_segment(ai_car)

        loops = 100 if (hyperspeed and mode == "TRAINING") else 1

        for _ in range(loops):
            if mode == "MANUAL":
                player_car.update_manual()
                player_car.check_collision()
                if not player_car.alive: player_car.reset()

            elif mode == "TRAINING":
                manager.update(ai_car)

        car_to_draw = player_car if mode == "MANUAL" else ai_car
        draw(WIN, images, car_to_draw, mode, manager, hyperspeed)

    manager.save_model()
    pygame.quit()

if __name__ == "__main__":
    main()
