import matplotlib.pyplot as plt
import numpy as np
import random
import pygame

days_to_split = 30
# number of days in a peak season multiplied by number of years queen bee of original hive lives
queen_bee_lifetime = 60 * 3

width = 1000
height = 1000
centre = (width/2, height/2)
pygame.init()
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()

t = 30
radius = 50
angles = {0:[0]}

screen.fill((0,0,0))
pygame.draw.circle(screen, (255,0,0), centre, 4)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.circle(screen, (255, 255, 255), centre, radius, 1)
    new_angles = []
    for r in angles.keys():
        for angle in angles[r]:
            max_angle = angle + (np.pi / 4)
            min_angle = angle - (np.pi / 4)
            angle2 = random.uniform(min_angle, max_angle)
            x1 = r * np.cos(angle)
            y1 = r * np.sin(angle)
            x2 = radius * np.cos(angle2)
            y2 = radius * np.sin(angle2)
            pygame.draw.circle(screen, (255,0,0), (centre[0]+x2, centre[1]+y2), 4)
            pygame.draw.line(screen, (128,128,128), (centre[0]+x1,centre[1]+y1), (centre[0]+x2,centre[1]+y2), 1)
            new_angles.append(angle2)

    angles[radius] = new_angles
    if t % 60 == 0:
        radius += 50
    radius += 25
    t += days_to_split
    pygame.display.flip()
    clock.tick(1)
    print(t)

    if t > queen_bee_lifetime:
        running = False

plt.show()