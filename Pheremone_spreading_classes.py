import random
import pygame
import math
import numpy as np

class BeeGroup:
    def __init__(self, name, color, population, radius_range, centre):
        self.name = name
        self.color = color
        self.population = population
        self.radius_range = radius_range
        self.positions = self.generate_positions(centre)

    # generate the original positions for each group of bees
    def generate_positions(self, centre):
        theta = np.random.uniform(0, 2 * np.pi, self.population)
        u = np.random.uniform(0, 1, self.population)
        r = np.sqrt(self.radius_range[0] ** 2 + (self.radius_range[1] ** 2 - self.radius_range[0] ** 2) * u)
        x = centre[0] + (r * np.cos(theta))
        y = centre[1] + (r * np.sin(theta))
        return list(zip(x, y))

    # find the distance from the queen bee
    def dist_from_queen(self, centre, x, y):
        return math.sqrt((centre[0] - x) ** 2 + (centre[1] - y) ** 2)

    # normalize the direction to queen bee
    def norm_direction(self, centre, x, y):
        dist = BeeGroup.dist_from_queen(self, centre, x, y)
        if dist != 0:
            ux = (centre[0] - x) / dist
            uy = (centre[1] - y) / dist
        else:
            ux, uy = 0, 0
        return ux, uy

    # plot the bees on the screen
    def plot_bees(self, screen):
        for x, y in self.positions:
            pygame.draw.circle(screen, self.color, (int(x), int(y)), 2)

class Retinue(BeeGroup):
    def update_pos(self, queen_pos, orbit_speed=0.03, attraction_strength=0.1, noise=0.7, r_min=5, r_max=15):
        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

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
            oy = ux * orbit_speed

            # random motion
            nx = random.uniform(-noise, noise)
            ny = random.uniform(-noise, noise)

            # new x,y pos
            x_new = x + ax + ox + nx
            y_new = y + ay + oy + ny

            self.positions[i] = (x_new, y_new)


class Nurse(BeeGroup):
    def update_pos(self, queen_pos, brood_radius=50, attraction_strength=0.01, noise=0.7):
        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

            # attraction to the centre of the hive
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

            self.positions[i] = (x_new, y_new)

class OtherBees(BeeGroup):
    def update_pos(self, queen_pos, attraction_strength, noise):
        r_min = self.radius_range[0]
        r_max = self.radius_range[1]

        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

            if dist > r_max:
                ax = ux * attraction_strength
                ay = uy * attraction_strength
            elif dist < r_min:
                ax = -ux * attraction_strength
                ay = -uy * attraction_strength
            else:
                ax, ay = 0, 0

            # random walk
            nx = random.uniform(-noise, noise)
            ny = random.uniform(-noise, noise)

            # New position
            x_new = x + ax + nx
            y_new = y + ay + ny

            self.positions[i] = (x_new, y_new)


class Simulation(BeeGroup):
    def __init__(self, width=1000, height=1000):
        self.width = width
        self.height = height
        centre = (width/2, height/2)

        population = {'queen': 1, 'retinue': 60, 'nurse': 450, 'in_hive_workers': 540, 'drones': 150}
        self.bee_groups = [
            BeeGroup('queen', (255,255,0), population['queen'], [0,0], centre),
            Retinue('retinue', (255,150,0), population['retinue'], [5,15], centre),
            Nurse('nurse', (0,255,0), population['nurse'], [20,100], centre),
            OtherBees('in_hive_workers', (0,255,255), population['in_hive_workers'], [100,300], centre),
            OtherBees('drones', (0,128,255), population['drones'], [200,400], centre)
        ]

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((0,0,0))
            queen_pos = self.bee_groups[0].positions[0]

            for group in self.bee_groups[1:3]:
                group.update_pos(queen_pos)
                group.plot_bees(self.screen)

            self.bee_groups[0].plot_bees(self.screen)
            self.bee_groups[3].update_pos(queen_pos, 0.004, 2.5)
            self.bee_groups[3].plot_bees(self.screen)
            self.bee_groups[4].update_pos(queen_pos, 0.002, 1.0)
            self.bee_groups[4].plot_bees(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()

