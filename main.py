import pygame
import time
import math
from utils import scale_image, blit_rotate_center

# --- CONFIG ---
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

# The Border is crucial for collision
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Simulation")

FPS = 60

class Car:
    IMG = RED_CAR
    START_POS = (180, 200) # Moved this to the parent class for easier access

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

    def move_backward(self): # Added backward movement
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def reduce_speed(self): # Added friction
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
            print("CRASH! Car Died.") # Added print for debugging

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

# Renamed to PlayerCar to reflect it's not AI yet
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

# Use PlayerCar instead of AICar
cars = [PlayerCar(4, 4)] # Reduced speed slightly for easier control

while run:
    clock.tick(FPS)

    draw(WIN, images, cars)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    for car in cars:
        if car.alive:
            car.update_manual() # Call the manual control method

pygame.quit()
