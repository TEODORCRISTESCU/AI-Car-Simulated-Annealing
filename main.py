import pygame
import math
from utils import scale_image, blit_rotate_center
import random

# --- CONFIG ---
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

# The Border is crucial for collision
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)
START_POS = (180, 200)
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Simulation")

FPS = 60

# ======== CAR CLASSES ========

class Car:
    IMG = RED_CAR
    START_POS = (180, 200)

    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.alive = True

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

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
            print("CRASH! Car Died.")

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi


class PlayerCar(Car):
    def update_manual(self):
        keys = pygame.key.get_pressed()
        moved = False

        if keys[pygame.K_a]:
            self.rotate(left=True)
        if keys[pygame.K_d]:
            self.rotate(right=True)
        if keys[pygame.K_w]:
            moved = True
            self.move_forward()
        if keys[pygame.K_s]:
            moved = True
            self.move_backward()

        if not moved:
            self.reduce_speed()

# =============== SIMULATED ANNEALING ===============

ACTIONS = ["ACCELERATE", "ROTATE_LEFT", "ROTATE_RIGHT", "GO_STRAIGHT"]

ACTION_REPEAT = 5

CHECKPOINTS = None # we need to implement a load of checkpoints cuz this map is massive
# Click on the track to see some coordinates to make this easier
def apply_action(car, action):
    if action == "ACCELERATE":
        car.move_forward()

    elif action == "BRAKE":
        car.move_backward()

    elif action == "GO_STRAIGHT":
        if car.vel > 0:
            car.reduce_speed()
        else:
            car.vel = 0
            car.move()

    elif action == "ROTATE_LEFT":
        car.rotate(left=True)
        car.move()

    elif action == "ROTATE_RIGHT":
        car.rotate(right=True)
        car.move()

def get_checkpoint_index(car): # Return the index of the closest checkpoint within a radius.

    cx, cy = car.x, car.y
    for i, (px, py) in enumerate(CHECKPOINTS):
        if math.hypot(cx - px, cy - py) < 40:  # radius threshold
            return i
    return -1

def random_actions(length=500):
    return [random.choice(ACTIONS) for _ in range(length)]

def measure_progress(car, last_checkpoint_idx):
    # REWARD: reaching next checkpoint
    idx = get_checkpoint_index(car)
    reward = 0

    if idx > last_checkpoint_idx:
        reward += 200  # reward for reaching a new checkpoint
        last_checkpoint_idx = idx

    # EXTRA: distance toward next checkpoint
    if last_checkpoint_idx + 1 < len(CHECKPOINTS):
        next_x, next_y = CHECKPOINTS[last_checkpoint_idx + 1]
        dist = math.hypot(car.x - next_x, car.y - next_y)
        reward += max(0, 100 - dist * 0.5)

    if car.vel < 0:
        reward -= 50

    if not car.alive:
        reward -= 500

    return reward, last_checkpoint_idx

def simulate(action_sequence):
    car = Car(4, 4)
    total_reward = 0
    last_checkpoint_idx = 0

    for action in action_sequence:
        for _ in range(ACTION_REPEAT):

            if not car.alive:
                total_reward -= 500
                return total_reward

            apply_action(car, action)

            r, last_checkpoint_idx = measure_progress(car, last_checkpoint_idx)
            total_reward += r

    return total_reward

def neighbour(solution):
    new = solution[:]
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(solution)-1)
        new[idx] = random.choice(ACTIONS)
    return new

def simulated_annealing():
    T = 1000
    cooling = 0.995
    current = random_actions()
    current_score = simulate(current)
    best = current
    best_score = current_score

    while T > 1:
        candidate = neighbour(current)
        candidate_score = simulate(candidate)

        if candidate_score > current_score:
            current, current_score = candidate, candidate_score
        else:
            probability = math.exp((candidate_score - current_score) / T)
            if random.random() < probability:
                current, current_score = candidate, candidate_score

        if candidate_score > best_score:
            best, best_score = candidate, candidate_score

        T *= cooling

    return best, best_score

def draw(win, images, cars):
    for img, pos in images:
        win.blit(img, pos)

    for car in cars:
        if car and car.alive:
            car.draw(win)

    pygame.display.update()

# =============== MAIN LOOP WITH MODES ===============

run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

mode = "manual"  # or "AI"
player_car = PlayerCar(4, 4)
ai_car = None
replay_actions = []
replay_step = 0

while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                mode = "manual"
                player_car = PlayerCar(4, 4)
                print("Switched to manual control.")

            if event.key == pygame.K_s and mode != "AI":
                print("Running simulated annealing (this may freeze UI until done)")
                best, score = simulated_annealing()
                print("Best score:", score)
                replay_actions = best
                replay_step = 0
                ai_car = Car(4, 4)
                mode = "AI"
                print("Switched to AI replay.")

    if mode == "manual":
        player_car.update_manual()
        draw(WIN, images, [player_car])

    elif mode == "AI":
        if ai_car and replay_step < len(replay_actions) and ai_car.alive:
            action = replay_actions[replay_step]
            apply_action(ai_car, action)
            replay_step += 1
        draw(WIN, images, [ai_car])

pygame.quit()
