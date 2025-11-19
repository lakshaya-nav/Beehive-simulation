import random
import pygame
import math
import numpy as np

WIDTH = 1000
HEIGHT = 1000

def generate_original_pos(pop, radius, positions):
    keys = list(positions.keys())
    for i in range(len(radius)):
        theta = np.random.uniform(0, 2*math.pi, pop[i])
        u = np.random.uniform(0, 1, pop[i])
        r = np.sqrt(radius[i][0]**2 + (radius[i][1]**2 - radius[i][0]**2)*u)

        x = WIDTH/2 + (r * np.cos(theta))
        y = HEIGHT/2 + (r * np.sin(theta))
        positions[keys[i+1]][1].extend(list(zip(x,y)))
    return positions

# find the distance from the queen bee
def dist_from_queen(cx, cy, x, y):
    return math.sqrt((cx-x)**2 + (cy-y)**2)

# normalize the direction to queen bee
def norm_direction(cx, cy, x, y):
    dist = dist_from_queen(cx, cy, x, y)
    if dist != 0:
        ux = (cx - x) / dist
        uy = (cy - y) / dist
    else:
        ux, uy = 0, 0
    return ux, uy

# movement of the retinue resembles an orbital behaviour around the queen bee
def update_retinue_pos(positions, radius, orbit_speed=0.03, attraction_strength=0.1, noise=0.7):
    queen_x, queen_y = positions['queen'][1][0]
    r_min = radius[0][0]
    r_max = radius[0][1]

    for i,(x, y) in enumerate(positions['retinue'][1]):
        dist = dist_from_queen(queen_x, queen_y, x, y)
        ux, uy = norm_direction(queen_x, queen_y, x, y)

        # attraction force of the retinue to the queen bee
        if dist > r_max:
            ax = ux * attraction_strength
            ay = uy * attraction_strength
        elif dist < r_min:
            ax = -ux * attraction_strength
            ay = -uy * attraction_strength
        else:
            ax, ay = 0, 0

        # orbit around the queen bee (perpendicular to queen direction)
        ox = -uy * orbit_speed
        oy =  ux * orbit_speed

        # random motion
        nx = random.uniform(-noise, noise)
        ny = random.uniform(-noise, noise)

        # new x,y pos
        x_new = x + ax + ox + nx
        y_new = y + ay + oy + ny

        positions['retinue'][1][i] = (x_new,y_new)

def update_nurse_pos(positions, brood_radius, attraction_strength = 0.01, noise=0.7):
    queen_x, queen_y = positions['queen'][1][0]
    for i, (x, y) in enumerate(positions['nurse'][1]):
        dist = dist_from_queen(queen_x, queen_y, x, y)
        ux, uy = norm_direction(queen_x, queen_y, x, y)

        # attraction to centre of hive
        if dist > brood_radius:
            ax = ux * attraction_strength
            ay = uy * attraction_strength
        else:
            ax = -ux * attraction_strength
            ay = -uy * attraction_strength

        # random motion
        nx = random.uniform(-noise, noise)
        ny = random.uniform(-noise, noise)

        # new x,y pos
        x_new = x + ax + nx
        y_new = y + ay + ny

        positions['nurse'][1][i] = (x_new, y_new)

def update_in_hive_workers_pos(positions, attraction=0.004, noise=2.5):
    queen_x, queen_y = positions['queen'][1][0]
    r_min = 100
    r_max = 300

    for i, (x, y) in enumerate(positions['in_hive_workers'][1]):
        dist = dist_from_queen(queen_x, queen_y, x, y)
        ux, uy = norm_direction(queen_x, queen_y, x, y)

        if dist > r_max:
            ax = ux * attraction
            ay = uy * attraction
        elif dist < r_min:
            ax = -ux * attraction
            ay = -uy * attraction
        else:
            ax, ay = 0, 0

        # random walk
        nx = random.uniform(-noise, noise)
        ny = random.uniform(-noise, noise)

        # New position
        x_new = x + ax + nx
        y_new = y + ay + ny

        positions['in_hive_workers'][1][i] = (x_new, y_new)

def update_drones_pos(positions, attraction=0.002, noise=1.0):
    queen_x, queen_y = positions['queen'][1][0]
    r_min = 200
    r_max = 500

    for i, (x, y) in enumerate(positions['drones'][1]):
        dist = dist_from_queen(queen_x, queen_y, x, y)
        ux, uy = norm_direction(queen_x, queen_y, x, y)

        if dist > r_max:
            ax = -ux * attraction
            ay = -uy * attraction
        elif dist < r_min:
            ax = ux * attraction
            ay = uy * attraction
        else:
            ax, ay = 0, 0

        # Random slow motion
        nx = random.uniform(-noise, noise)
        ny = random.uniform(-noise, noise)

        x_new = x + ax + nx
        y_new = y + ay + ny

        positions['drones'][1][i] = (x_new, y_new)



def plot_pos(positions, screen):
    for key in list(positions.keys()):
        points = positions[key][1]
        colour = positions[key][0]
        for x, y in points:
            pygame.draw.circle(screen, colour, (int(x), int(y)), 2)

def run_simulation():
    positions = {'queen': [[255, 255, 0], [(WIDTH / 2, HEIGHT / 2)]], 'retinue': [[255, 150, 0], []],
                 'nurse': [[0, 255, 0], []], 'in_hive_workers': [[0, 255, 255], []],
                 'drones': [[0, 128, 255], []]}
    total_pop = 3000
    pop_percentage = [0.02, 0.15, 0.18, 0.05]
    pop = [int(x * total_pop) for x in pop_percentage]
    radius = [[5, 15], [20, 100], [100, 300], [200, 400]]

    positions = generate_original_pos(pop, radius, positions)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        update_retinue_pos(positions, radius)
        update_nurse_pos(positions, 50)
        update_in_hive_workers_pos(positions)
        update_drones_pos(positions)
        plot_pos(positions, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    run_simulation()