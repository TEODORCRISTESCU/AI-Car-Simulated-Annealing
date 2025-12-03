import pygame
import math
import random
import pickle
import os
import sys
from utils import scale_image, blit_rotate_center

# --- CONFIG & ASSETS ---
pygame.font.init()
STAT_FONT = pygame.font.SysFont("consolas", 18)

try:
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
pygame.display.set_caption("AI Trainer: Continuous Driving")

FPS = 60

# --- CONSTANTS ---
ACTIONS = ["ACCELERATE", "BRAKE", "ROTATE_LEFT", "ROTATE_RIGHT", "GO_STRAIGHT"]
ACTION_REPEAT = 5
CHECKPOINTS = [
    (116, 71), (49, 138), (70, 481), (315, 732),
    (408, 671), (429, 499), (571, 504), (611, 705),
    (727, 713), (727, 386), (438, 360), (410, 287),
    (721, 242), (726, 90), (298, 81), (280, 362),
    (209, 401), (178, 359)
]

# ======== CAR CLASS ========

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

# ======== AI MANAGER ========

class TrainingManager:
    def __init__(self):
        # -- Persistence State --
        self.committed_actions = []
        self.committed_path_points = [(180, 200)]
        self.start_state = None
        self.start_checkpoint_idx = 0

        # -- Annealing State --
        self.current_segment_actions = self.random_actions(150)
        self.best_segment_actions = list(self.current_segment_actions)
        self.best_segment_score = -99999
        self.accepted_score = -99999

        # -- Simulation State --
        self.step_index = 0
        self.frame_counter = 0
        self.current_reward = 0
        self.current_path_points = []

        # -- Tuning --
        self.INITIAL_TEMP = 800
        self.T = self.INITIAL_TEMP
        self.cooling = 0.992

        self.load_model()

    def random_actions(self, length):
        return [random.choice(ACTIONS) for _ in range(length)]

    def load_model(self):
        if os.path.exists("saved_state.pkl"):
            try:
                with open("saved_state.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.committed_actions = data["committed_actions"]
                    self.committed_path_points = data.get("committed_path", [(180, 200)])
                    self.start_state = data["start_state"]
                    self.start_checkpoint_idx = data["start_checkpoint_idx"]
                    print("Hot Resume: Loaded previous state.")
            except:
                print("Could not load save file. Starting fresh.")

    def save_model(self):
        data = {
            "committed_actions": self.committed_actions,
            "committed_path": self.committed_path_points,
            "start_state": self.start_state,
            "start_checkpoint_idx": self.start_checkpoint_idx
        }
        with open("saved_state.pkl", "wb") as f:
            pickle.dump(data, f)

    def reset_car_to_segment_start(self, car):
        if self.start_state:
            car.set_state(self.start_state)
        else:
            car.reset()

        self.step_index = 0
        self.frame_counter = 0
        self.current_reward = 0
        self.current_path_points = []

    def commit_segment(self, car):
        """We reached a checkpoint! Lock in these actions."""
        # 1. Save committed actions
        actions_taken = self.current_segment_actions[:self.step_index+1]
        self.committed_actions.extend(actions_taken)
        self.committed_path_points.extend(self.current_path_points)

        # 2. Update Start State for next segment
        self.start_state = car.get_state()
        self.start_checkpoint_idx = (self.start_checkpoint_idx + 1) % len(CHECKPOINTS)

        print(f"Checkpoint {self.start_checkpoint_idx} Reached! Boosting Temp.")

        # 3. CONTINUITY LOGIC (Soft Reset)
        # Instead of random noise, we initialize the NEXT segment with
        # a continuation of the previous best actions (momentum)

        # We start with mostly "GO_STRAIGHT" or "ACCELERATE" to encourage flow
        # then let annealing mutate it.
        fresh_actions = ["ACCELERATE"] * 150

        self.current_segment_actions = fresh_actions
        self.best_segment_actions = list(self.current_segment_actions)
        self.best_segment_score = -99999
        self.accepted_score = -99999

        # 4. SMART TEMP BOOST
        # Increase temperature by 250, but cap it at INITIAL_TEMP
        # This gives it enough "heat" to learn the new turn, but not enough to go crazy
        self.T = min(self.INITIAL_TEMP, self.T + 250)

        self.save_model()

    def prepare_next_attempt(self, car):
        score = self.current_reward

        # Annealing Acceptance
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
                self.best_segment_actions = list(self.current_segment_actions)
            else:
                self.current_segment_actions = list(self.best_segment_actions)

        self.mutate_actions()

        # Cooling
        self.T = max(10, self.T * self.cooling)

        self.reset_car_to_segment_start(car)

    def mutate_actions(self):
        # Dynamic mutation rate based on Temp
        mutation_rate = 1
        if self.T > 400: mutation_rate = 5
        elif self.T > 100: mutation_rate = 2

        for _ in range(random.randint(1, mutation_rate)):
            idx = random.randint(0, len(self.current_segment_actions)-1)
            self.current_segment_actions[idx] = random.choice(ACTIONS)

    def update(self, car):
        if not car.alive:
            self.current_reward -= 500
            self.prepare_next_attempt(car)
            return

        if self.step_index >= len(self.current_segment_actions):
            self.current_reward -= 100
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

        if self.frame_counter == 0:
            self.current_path_points.append((car.x + car.img.get_width()/2, car.y + car.img.get_height()/2))

        # Reward
        target = CHECKPOINTS[self.start_checkpoint_idx]
        dist = math.hypot(car.x - target[0], car.y - target[1])

        self.current_reward += (1000 / (dist + 1))
        self.current_reward -= 1

        if car.vel < 0: self.current_reward -= 5
        elif car.vel < 0.5: self.current_reward -= 2

        if dist < 40:
            self.commit_segment(car)
            self.reset_car_to_segment_start(car)
            return

        self.frame_counter += 1
        if self.frame_counter >= ACTION_REPEAT:
            self.step_index += 1
            self.frame_counter = 0

# ======== VISUALIZATION & MAIN ========

def draw_paths(win, manager):
    if len(manager.committed_path_points) > 1:
        pygame.draw.lines(win, (0, 255, 0), False, manager.committed_path_points, 3)
    if len(manager.current_path_points) > 1:
        pygame.draw.lines(win, (255, 50, 50), False, manager.current_path_points, 2)

def draw_ui(win, mode, manager, hyperspeed):
    bg_rect = pygame.Rect(0, HEIGHT - 140, 320, 140)
    pygame.draw.rect(win, (0, 0, 0), bg_rect)
    pygame.draw.rect(win, (255, 255, 255), bg_rect, 2)

    speed_txt = "HYPERSPEED" if hyperspeed else "Normal Speed"
    col = (255, 100, 100) if hyperspeed else (200, 200, 200)

    lines = [
        f"Mode: {mode} | {speed_txt}",
        f"Target CP: {manager.start_checkpoint_idx}",
        f"Temp: {manager.T:.1f} (Boost on CP)",
        f"Current Reward: {int(manager.current_reward)}",
        f"Segs Solved: {len(manager.committed_actions)//5}",
        "KEYS: [T]rain, [M]anual, [H]yperspeed"
    ]

    for i, line in enumerate(lines):
        txt = STAT_FONT.render(line, 1, col if i == 0 else (255, 255, 255))
        win.blit(txt, (10, HEIGHT - 130 + (i * 20)))

def draw(win, images, car, mode, manager, hyperspeed):
    for img, pos in images:
        win.blit(img, pos)

    if mode == "TRAINING":
        draw_paths(win, manager)
        target = CHECKPOINTS[manager.start_checkpoint_idx]
        pygame.draw.circle(win, (0, 255, 255), target, 10)

    if car.alive:
        car.draw(win)

    draw_ui(win, mode, manager, hyperspeed)
    pygame.display.update()

def main():
    run = True
    clock = pygame.time.Clock()

    images = [(TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

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
