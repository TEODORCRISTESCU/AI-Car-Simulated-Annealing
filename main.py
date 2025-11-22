import pygame
import time
import math
from utils import scale_image, blit_rotate_center

# --- CONFIG ---
# Reduce image size slightly to fit more tracks if needed
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

# The Border is crucial for collision
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

# Using the Red Car for our AI
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genetic Mutation Simulation")

FPS = 60


class Car:
    IMG = RED_CAR

    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

        # Simulation State
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

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

        # Step B: Collision Check
        self.check_collision()

    def check_collision(self):
        # If we hit the border, we die
        if self.collide(TRACK_BORDER_MASK) is not None:
            self.alive = False

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi


class AICar(Car):
    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.radars = []

    START_POS = (180, 200)  # Starting position on the TechWithTim Track

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)

    def update(self):
        # Step A: Physics Loop
        # For now, just drive forward and spin to prove we exist
        self.move_forward()
        self.rotate(left=True)


# --- MAIN SIMULATION LOOP ---
def draw(win, images, cars):
    for img, pos in images:
        win.blit(img, pos)

    for car in cars:
        if car.alive:
            car.draw(win)

    pygame.display.update()


run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

# We will eventually have a list of 50+ cars here
cars = [AICar(4, 4)]

while run:
    clock.tick(FPS)

    draw(WIN, images, cars)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    for car in cars:
        if car.alive:
            car.update()

pygame.quit()