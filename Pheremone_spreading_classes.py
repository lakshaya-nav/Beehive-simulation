import random
import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree


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
    def plot_bees(self, screen, pheromones, offset):
        for i, (x,y) in enumerate(self.positions):
            bee_index = offset + i
            if pheromones[bee_index] > 0:
                pygame.draw.circle(screen, (255,255,255), (int(x), int(y)), 2)
            else:
                pygame.draw.circle(screen, self.color, (int(x), int(y)), 2)

class Retinue(BeeGroup):
    ATTRACTION_STRENGTH = 0.1
    NOISE = 0.7
    ORBIT_SPEED = 0.03
    R_MIN = 5
    R_MAX = 15

    def update_pos(self, queen_pos):
        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

            # attraction force of the retinue to the queen bee
            if dist > Retinue.R_MAX:
                ax = ux * Retinue.ATTRACTION_STRENGTH
                ay = uy * Retinue.ATTRACTION_STRENGTH
            elif dist < Retinue.R_MIN:
                ax = -ux * Retinue.ATTRACTION_STRENGTH
                ay = -uy * Retinue.ATTRACTION_STRENGTH
            else:
                ax, ay = 0, 0

            # orbit around the queen bee (perpendicular to queen direction)
            ox = -uy * Retinue.ORBIT_SPEED
            oy = ux * Retinue.ORBIT_SPEED

            # random motion
            nx = random.uniform(-Retinue.NOISE, Retinue.NOISE)
            ny = random.uniform(-Retinue.NOISE, Retinue.NOISE)

            # new x,y pos
            x_new = x + ax + ox + nx
            y_new = y + ay + oy + ny

            self.positions[i] = (x_new, y_new)


class Nurse(BeeGroup):
    ATTRACTION_STRENGTH = 0.01
    NOISE = 0.7
    BROOD_RADIUS = 50

    def update_pos(self, queen_pos):
        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

            # attraction to the centre of the hive
            if dist > Nurse.BROOD_RADIUS:
                ax = ux * Nurse.ATTRACTION_STRENGTH
                ay = uy * Nurse.ATTRACTION_STRENGTH
            else:
                ax = -ux * Nurse.ATTRACTION_STRENGTH
                ay = -uy * Nurse.ATTRACTION_STRENGTH

            # random motion
            nx = random.uniform(-Nurse.NOISE, Nurse.NOISE)
            ny = random.uniform(-Nurse.NOISE, Nurse.NOISE)

            # new x,y pos
            x_new = x + ax + nx
            y_new = y + ay + ny

            self.positions[i] = (x_new, y_new)

class OtherBees(BeeGroup):
    ATTRACTION_STRENGTH = 0.004
    NOISE = 2.5

    def update_pos(self, queen_pos):
        r_min = self.radius_range[0]
        r_max = self.radius_range[1]

        for i, (x, y) in enumerate(self.positions):
            dist = self.dist_from_queen(queen_pos, x, y)
            ux, uy = self.norm_direction(queen_pos, x, y)

            if dist > r_max:
                ax = ux * self.ATTRACTION_STRENGTH
                ay = uy * self.ATTRACTION_STRENGTH
            elif dist < r_min:
                ax = -ux * self.ATTRACTION_STRENGTH
                ay = -uy * self.ATTRACTION_STRENGTH
            else:
                ax, ay = 0, 0

            # random walk
            nx = random.uniform(-self.NOISE, self.NOISE)
            ny = random.uniform(-self.NOISE, self.NOISE)

            # New position
            x_new = x + ax + nx
            y_new = y + ay + ny

            self.positions[i] = (x_new, y_new)

class WorkerBees(OtherBees):
    ATTRACTION_STRENGTH = 0.004
    NOISE = 2.5

class Drone(OtherBees):
    ATTRACTION_STRENGTH = 0.002
    NOISE = 1.0

class PheromoneSpreading:
    def __init__(self, positions):
        self.queen_lifetime = 3
        self.alpha = 1.5
        self.beta = 0.0025
        self.p_init = 130
        self.positions = positions
        self.graph = nx.Graph()
        self.pheromone_per_bee = [0] * len(self.positions)
        self.pheromone_per_bee[0] = self.p_init
        self.transfer_rate = 0.01

    def pheromone_calc(self, t):
        p = self.p_init*math.exp(-self.alpha*(1-math.exp(-self.beta * t)))
        return p

    def plot_pheromone_levels(self):
        days = self.queen_lifetime * 365
        t = []
        p = []
        for i in range(days):
            t.append(i)
            p.append(self.pheromone_calc(i))

        plt.plot(t,p)
        plt.xlabel("Queen age (days)")
        plt.ylabel("Pheromone level")
        plt.title("Queen pheromone decline over time")
        plt.grid(True)
        plt.show()

    def update_pheromones(self, threshold=3):
        tree = KDTree(self.positions)
        pairs = tree.query_pairs(r=threshold)  # only return pairs within threshold

        for i, pos in enumerate(self.positions):
            self.graph.add_node(i, pos=pos)

        for i, j in pairs:
            dist = math.dist(self.positions[i], self.positions[j])
            self.graph.add_edge(i, j, weight=1 / dist)

        updates = [0] * len(self.pheromone_per_bee)
        for i, p in enumerate(self.pheromone_per_bee):
            if p > 0:
                neighbors = list(self.graph.neighbors(i))
                for n in neighbors:
                    updates[n] += p * self.transfer_rate * self.graph[i][n]['weight']

        for i in range(len(self.pheromone_per_bee)):
            self.pheromone_per_bee[i] += updates[i]



class Simulation:
    def __init__(self, width=1000, height=1000):
        self.width = width
        self.height = height
        centre = (width/2, height/2)
        population = {'queen': 1, 'retinue': 60, 'nurse': 450, 'in_hive_workers': 540, 'drones': 150}
        self.queen = BeeGroup('queen', (255,255,0), population['queen'], [0,0], centre)
        self.retinues = Retinue('retinue', (255,150,0), population['retinue'], [5,15], centre)
        self.nurses = Nurse('nurse', (0,255,0), population['nurse'], [20,100], centre)
        self.workers = WorkerBees('in_hive_workers', (0, 255, 255), population['in_hive_workers'], [100, 300], centre)
        self.drones = Drone('drones', (0, 128, 255), population['drones'], [200, 400], centre)

        bees_positions = (self.queen.positions + self.retinues.positions + self.nurses.positions + self.workers.positions + self.drones.positions)
        self.pheromones = PheromoneSpreading(bees_positions)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def quit(self):
        total = 0
        for i in self.pheromones.pheromone_per_bee:
            if i > 0:
                total += 1

        if total > 0.98 * len(self.pheromones.pheromone_per_bee):
            return True
        return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((0,0,0))
            queen_pos = self.queen.positions[0]
            # plot queen bee
            index = 0
            self.queen.plot_bees(self.screen, self.pheromones.pheromone_per_bee, index)
            index += len(self.queen.positions)
            for group in [self.retinues, self.nurses, self.workers, self.drones]:
                group.update_pos(queen_pos)
                group.plot_bees(self.screen, self.pheromones.pheromone_per_bee, index)
                index += len(group.positions)

            all_positions = (self.queen.positions + self.retinues.positions + self.nurses.positions + self.workers.positions + self.drones.positions)
            self.pheromones.positions = all_positions
            self.pheromones.update_pheromones()

            if self.quit():
                running = False

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
