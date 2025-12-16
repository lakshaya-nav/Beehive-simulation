import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random
import pygame
import math
import networkx as nx
from scipy.spatial import KDTree


class BeeGroup:
    def __init__(self, name, colour, population, radius_range, centre):
        self.name = name
        self.colour = colour
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
                pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 2)
            else:
                pygame.draw.circle(screen, self.colour, (int(x), int(y)), 2)

class PheromoneSpreading:
    def __init__(self, positions):
        self.queen_lifetime = 3
        self.alpha = 1.5
        self.beta = 0.0025
        self.p_init = 1
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
                    updates[n] += p * self.transfer_rate
                    updates[i] -= p * self.transfer_rate
        updates[0] += self.p_init

        for i in range(len(self.pheromone_per_bee)):
            self.pheromone_per_bee[i] += updates[i]

    def percent_worker_pheromone(self, start_idx, pop):
        avg = sum(self.pheromone_per_bee[start_idx:start_idx + pop]) / pop
        print(f'avg: {avg}')
        print(f'queen: {self.pheromone_per_bee[0]}')
        percent_of_queen_pheromone = (avg/self.pheromone_per_bee[0])*100
        return percent_of_queen_pheromone


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


class Simulation:
    def __init__(self, population, width=1000, height=1000):
        self.width = width
        self.height = height
        centre = (width / 2, height / 2)
        self.population = population
        self.queen = BeeGroup('queen', (255, 255, 0), self.population['queen'], [0, 0], centre)
        self.retinues = Retinue('retinue', (255, 150, 0), self.population['retinue'], [5, 15], centre)
        self.nurses = Nurse('nurse', (0, 255, 0), self.population['nurse'], [20, 100], centre)
        self.workers = WorkerBees('in_hive_workers', (0, 255, 255), self.population['in_hive_workers'], [100, 300],
                                  centre)
        self.drones = Drone('drones', (0, 128, 255), self.population['drones'], [200, 400], centre)

        bees_positions = (
                    self.queen.positions + self.retinues.positions + self.nurses.positions + self.workers.positions + self.drones.positions)
        self.pheromones = PheromoneSpreading(bees_positions)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def quit(self):
        total = 0
        for i in self.pheromones.pheromone_per_bee:
            if i > 0:
                total += 1

        if total > 0.95 * len(self.pheromones.pheromone_per_bee):
            return True
        return False


    def run(self):
        count = 0
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
            count += 1

        pygame.quit()
        print(count)
        start_idx = self.population['queen'] + self.population['retinue'] + self.population['nurse']
        return self.pheromones.percent_worker_pheromone(start_idx, self.population['in_hive_workers'])


def overcrowding(percent_of_queen_pheromone, total_pop, p):
    max_pop = p["maxCapacityBeehive"]
    min_pheromone = 0.001  # keep this constant unless you want to parametrize it too
    pop_weight = 0.5
    pheromone_weight = 0.5

    print("Printing overcrowding threshhold: ")
    print(p["overcrowding_threshhold"])
    print("Printing maximum capacity: ")
    print(max_pop)

    pop_factor = total_pop / max_pop

    if percent_of_queen_pheromone < min_pheromone:
        pheromone_factor = 1
    else:
        pheromone_factor = min_pheromone / percent_of_queen_pheromone

    if total_pop > max_pop:
        return 1

    overcrowding_index = (pop_weight * pop_factor) + (pheromone_weight * pheromone_factor)
    return overcrowding_index

# getting beehive populations based on total population
def beehive(population):
    without_queen = population-1 # exclude the queen
    pop = {
        "Q": 1,
        "E": round(without_queen*0.11), # eggs
        "L": round(without_queen*0.15), # larvae
        "Pp": round(without_queen*0.23), # pupae
        "W": round(without_queen*0.21), # workers = nurses
        "F": round(without_queen*0.28), # foragrers
        "D": round(without_queen*0.02), # drones
        }
    return pop

stateInit = beehive(40000) # getting initial state for population of 60000 bees
# print(stateInit)

params = {
    # Egg stage
    "alphaE": 0.33,    # per day
    "muE": 0.03,

    # Larval stage
    "alphaL": 0.17,
    "muL": 0.025,

    # Pupal stage
    "alphaP": 0.08,   # adult workers emerging - needs accurate tuning
    "muP": 0.003,

    # In-hive workers
    "worker_to_forager": 0.1, # worker_to_forager
    "muW": 0.02,

    # Foragers
    "muF": 0.15,

    # Drones
    "muD": 0.04,

    # workers split into retinue, nurses and remaining workers
    "retinuePercent": 0.05,
    "nursePercent": 0.3,
    "restOfWorkers": 0.65,

    # Swarm parameters
    "sigma": 0.55,                     # fraction of total adults that swarm
    "queen_loss_prob": 1.0,            # queen always leaves (for simplicity)
    "swarm_threshold": 30000,          # adult population threshold
    "swarm_steepness": 0.0002,         # steepness of logistic hazard
    "queenLossProb": 0.95, # probability queen leaves with the swarm
    "newQueenDelay": 2,   # days until new queen emerges

    "Wref": 1500, # refrence number for nurses that are required to care for fully func queens babies
    "maxEggs": 4000, # eggs a healthy and fully fed queen can lay
    "fertilized": 0.95, # percentage of eggs that get fertilized

    "maxCapacityBeehive": 80000,
    "overcrowding_threshhold": 0.8,

    # percentages of populations leaving when split happens
    "drone_percentage_leaving": 0.4,
    "swarm_leave_frac_workers": 0.70,   # higher: swarms are heavy on in-hive workers
    "swarm_leave_frac_foragers": 0.30,  # lower: fewer foragers leave
    "swarm_min_drones": 30,

}

def make_it_random(p_base):
    p = p_base.copy()

    # Egg stage
    p["alphaE"] = random.uniform(0.30, 0.36)      # ±10%
    p["muE"]    = random.uniform(0.02, 0.04)      # ±30%

    # Larval stage
    p["alphaL"] = random.uniform(0.15, 0.19)      # ±12%
    p["muL"]    = random.uniform(0.02, 0.03)      # ±20%

    # Pupal stage
    p["alphaP"] = random.uniform(0.07, 0.09)      # ±12%
    p["muP"]    = random.uniform(0.002, 0.004)    # ±30%

    # In-hive workers
    p["worker_to_forager"] = random.uniform(0.08, 0.12)  # ±20%
    p["muW"]               = random.uniform(0.015, 0.025)  # ±25%

    # Foragers
    p["muF"] = random.uniform(0.12, 0.18)         # ±20%

    # Drones
    p["muD"] = random.uniform(0.03, 0.05)         # ±25%

    # Worker caste ratios
    # we want them to add up to one:
    p["retinuePercent"] = random.uniform(0.045, 0.055)   # ±10%
    p["nursePercent"]   = random.uniform(0.24, 0.30)     # ±10%
    p["restOfWorkers"]  = 1 -  p["retinuePercent"] - p["nursePercent"] # so that they never exceed 1

    # Swarm parameters
    p["sigma"]            = random.uniform(0.50, 0.60)        # ±10%
    p["queen_loss_prob"]  = random.uniform(0.9, 1.0)          # near-certain
    p["swarm_threshold"]  = random.uniform(27000, 33000)      # ±10%
    p["swarm_steepness"]  = random.uniform(0.00015, 0.00025)  # ±20%
    p["queenLossProb"]    = random.uniform(0.90, 0.98)        # ±5%
    p["newQueenDelay"]    = random.randint(8, 12)              # 1–3 days

    # Queen laying / brood
    p["Wref"]      = random.uniform(1300, 1700)     # ±13%
    p["maxEggs"]   = random.uniform(3500, 4500)     # ±12%
    p["fertilized"] = random.uniform(0.92, 0.98)    # ±3%

    # Capacity & overcrowding
    p["maxCapacityBeehive"] = random.uniform(80000, 88000)  #
    p["overcrowding_threshhold"] = random.uniform(0.9, 0.95)    # 

    # Swarm split
    p["percentage_leaving"] = random.uniform(0.55, 0.65)   # ±8%
    p["drone_percentage_leaving"] = random.uniform(0.35, 0.4)
    return p


# turning dict to array (for ODEs)
def pack(state):
    return np.array([state["Q"], state["E"], state["L"], state["Pp"],
                     state["W"], state["F"], state["D"]])


#turnining into dict
def unpack(y):
    return {
        "Q": y[0],
        "E": y[1],
        "L": y[2],
        "Pp": y[3],
        "W": y[4],
        "F": y[5],
        "D": y[6],
    }

def q_lay(state, p, temperature=None):
    wref = p["Wref"]
    num_of_Nnurses = p["nursePercent"]*state["W"]
    nurse_factor = min(1.0, num_of_Nnurses / wref)

    if temperature == None:
        combined_factor = nurse_factor
    else:
        combined_factor = nurse_factor*temperature
    return combined_factor


def populationsForODEs(t, state, p, day_num, temperatures=None):
    #extracting information from our state to use

    Q, E, L, Pp, W, F, D = state

    # send temperature onyl if it is not None
    if temperatures is None:
        q = q_lay(unpack(state), p)
    else:
        q = q_lay(unpack(state), p, temperatures[day_num])

    # ODEs
    #for the eggs --> larvae --> pupae
    dE  = (p["maxEggs"]*Q*q- (p["alphaE"] + p["muE"]) * E)
    dL  = (p["alphaE"] * E - (p["alphaL"] + p["muL"]) * L)
    dPp = (p["alphaL"] * L - (p["alphaP"] + p["muP"]) * Pp)

    #out of the pupae that we have:
    #   95% turn into fertlized eggs -> workers and foragers
    dW = (p["fertilized"]*p["alphaP"] * Pp - p["worker_to_forager"] * W - p["muW"] * W)
    dF = (p["worker_to_forager"] * W - p["muF"] * F) # rho

    # the remaining 5% turns into drones
    dD = (p["alphaP"]*Pp*(1 - p["fertilized"]) - p["muD"]*D)

    # dQ is always 0 since the queen doesnt change increase or decrease
    dQ = 0

    return np.array([dQ, dE, dL, dPp, dW, dF, dD])


# making first beehive object with initial states
def add_to_history(state, t, num):
    # adds the previous beehive to the dictionary
    new_row = {
        "State": state,
        "Day of occurance": t,
        "Number of beehive": num
    }
    for col in hive_history:
        hive_history[col].append(new_row[col])


class Beehive():
    num_of_hives = 0
    def __init__(self, state, t):
        self.state = state
        self.starting_time = t
        Beehive.num_of_hives += 1
        self.which_hive = Beehive.num_of_hives # which number of beehive it is
        # adding to history
        add_to_history(self.state, self.starting_time, self.which_hive)

    def describe_hive(self):
        self.new = {"State": self.state,
               "Day of occurance": self.starting_time,
               "Number of beehive": self.which_hive}



#simulation loop
def simulate(Tmax, dt, initState, p_base, num_of_pher_sims=5, temperatures=None):
    global hive_history
    hive_history = {
        "State": [],
        "Day of occurance": [],
        "Number of beehive": []
    }
    hive1 = Beehive(stateInit, 0)
    overcrowding_index = 0.0

    # making array to track the crowding indexes of the hive so we dont have to calculate everyday
    overcrowding_index_history = []
    hives = {}
    t_current = 0
    state = initState.copy()

    # initializing queen_timer as None
    swarm_made = False
    queen_timer = None
    all_population_states = []
    total_pop = []

    while t_current < Tmax:
        print(t_current)
        print("\n")
        p = make_it_random(p_base)
        #p = p_base
        # solving odes to get populations everyday
        # --- always integrate dynamics (even if Q==0) ---
        if queen_timer is not None:
            state["Q"] = 0
        if temperatures is None:
            sol = solve_ivp(
                lambda t, y: populationsForODEs(t, y, p, t_current),
                [t_current, t_current + dt],
                pack(state),
            )
        else:
            sol = solve_ivp(
                lambda t, y: populationsForODEs(t, y, p, t_current, temperatures),
                [t_current, t_current + dt],
                pack(state),
            )

        y_float = sol.y[:, -1]
        y_new = np.round(y_float).astype(int)
        state = unpack(y_new)

        all_population_states.append(state)
        total_pop.append(sum(state.values()))

        print("Printing total population")
        print(len(total_pop))
        print(total_pop[-1])

        if t_current % num_of_pher_sims == 0:
            if state['Q'] == 1:
                if total_pop[-1] > 5000:
                    scale = 10  # or 20 if you need it faster
                    population = {
                        'queen': 1,
                        'retinue': max(1, int((state['W'] * p['retinuePercent']) / scale)),
                        'nurse': max(1, int((state['W'] * p['nursePercent']) / scale)),
                        'in_hive_workers': max(1, int((state['W'] * p['restOfWorkers']) / scale)),
                        'drones': max(1, int(state['D'] / scale)),
                    }
                    sim = Simulation(population)
                    p_workers = sim.run()
                    print(f'p_workers: {p_workers}')
                    overcrowding_index = overcrowding(p_workers, total_pop[-1], p)
                    print(f'Overcrowding index: {overcrowding_index}')
                    overcrowding_index_history.append(overcrowding_index)
                else:
                   print("Beehive depleted")
                   break
            else:
                print("queen died so making overcrowding index 0")
                overcrowding_index = 0
                overcrowding_index_history.append(overcrowding_index)

        print("overcrowding index: ")
        print(overcrowding_index)
        

        if state["Q"] == 1 and overcrowding_index > p["overcrowding_threshhold"]:
            print("Swarm triggered (overcrowding)")

            pre = state.copy()

            leaving = pre.copy()
            staying = pre.copy()

            # --- Queen goes with the swarm; parent becomes queenless ---
            leaving["Q"] = 1
            staying["Q"] = 0

            # --- Brood stays with parent; swarm has no brood ---
            leaving["E"] = leaving["L"] = leaving["Pp"] = 0
            staying["E"], staying["L"], staying["Pp"] = pre["E"], pre["L"], pre["Pp"]

            # --- Adults split (biased) ---
            leaving["W"] = int(round(pre["W"] * p["swarm_leave_frac_workers"]))
            leaving["F"] = int(round(pre["F"] * p["swarm_leave_frac_foragers"]))
            leaving["D"] = int(round(pre["D"] * p["drone_percentage_leaving"]))

            # Minimum drone floor (only if drones exist)
            if pre["D"] > 0:
                leaving["D"] = max(leaving["D"], min(p["swarm_min_drones"], pre["D"]))

            # --- Clamp leaving so it can never exceed what exists ---
            for k in ["W", "F", "D"]:
                leaving[k] = max(0, min(leaving[k], pre[k]))

            # --- Mass-conserving remainder ---
            staying["W"] = pre["W"] - leaving["W"]
            staying["F"] = pre["F"] - leaving["F"]
            staying["D"] = pre["D"] - leaving["D"]

            # Apply the parent state + start queen timer
            state = staying
            queen_timer = 0

            # Create the swarm hive
            numberOfHives = Beehive.num_of_hives
            hives[f"hive{numberOfHives + 1}"] = Beehive(leaving, t_current)

            overcrowding_index = 0
            overcrowding_index_history.append(overcrowding_index)




        # --- QUEENLESS PERIOD: parent waits for new queen to emerge ---
        if state["Q"] == 0:
            if queen_timer is None:
                queen_timer = 0
            else:
                queen_timer += dt

            # once the delay is over, the parent gets a new queen
            if queen_timer >= p["newQueenDelay"]:
                state["Q"] = 1
                queen_timer = None
                swarm_made = False
                print("New queen emerged in parent hive")

    
            
        t_current += 1

    to_return = pd.DataFrame(all_population_states)
    to_return["total"] = total_pop

    return to_return

def statistical_analysis(hive_history):

    #stats = pd.DataFrame(hive_history)
    stats = pd.DataFrame(hive_history)
    stats = stats.drop(index=0)
    #stats = pd.DataFrame(hive_history[1:])

    # getting means for all new beehives 
    populations = stats["State"].apply(pd.Series)

    # Compute mean of each population across all rows
    mean_populations = populations.mean()
    # compute standard deviation for each population across all rows

    std_populations = populations.std() 

    # getting mean num of days to split
    days_for_split = []
    #days_for_split[0] = stats["Day of occurance"][0]
    for i in range(1, len(stats)):
       # getting the distance in days per splits
       days_for_split.append(
            stats["Day of occurance"].iloc[i] -
            stats["Day of occurance"].iloc[i-1]
        )
       
    # saving to file
    
    # getting mean of that
    mean_split_days = np.array(days_for_split).mean()
    # getting standard deviation
    std_split_days = np.array(days_for_split).std()

    #print("Printing mean days_for_split")
    #print(mean_split_days)
    return mean_populations, std_populations, mean_split_days, std_split_days

def plot_without_total(values, x):

    plt.figure(figsize=(10, 6))
    values = values.drop(columns=["total"])
    print("Printing value of not_total")
    print("Printing columns of values")
    print(values.columns)
    for col in values.columns:
        plt.plot(x, values[col], label=col)

    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title("Beehive Population Dynamics Over Time")
    plt.legend(["Queen", "Egg Population", "Larave Population", "Pupae Population",
                "Worker Population", "Foragers Population", "Drones Population",
                "Total population"])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # extracting the values
    dt = 1

    print(stateInit)
    num_of_pher_sims = 15 
    values = simulate(60, dt, stateInit, params, num_of_pher_sims, temperatures=None)

    Evals = values["E"]
    Lvals = values["L"]
    Ppvals = values["Pp"]
    Wvals = values["W"]
    Fvals = values["F"]
    Dvals = values["D"]
    totalvals = values["total"]
    
    # statistics
    print("Pritning hive history")
    print(pd.DataFrame(hive_history))
    meanPops, stdPops, meanDays, stdDays = statistical_analysis(hive_history)
    print("Printing statistical analysis")
    print(f"Mean for each population: {meanPops}, std for populations: {stdPops}, mean days per split {meanDays}, stad days per split: {stdDays}")
    #statistical_analysis(hive_history)

    x = [x for x in range(len(Evals))]
    # plotting with and without total population using not_total
    
    plt.figure(figsize=(10, 6))

    for col in values.columns:
        plt.plot(x, values[col], label=col)

    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title("Beehive Population Dynamics Over Time")
    plt.legend(["Queen", "Egg Population", "Larave Population", "Pupae Population",
                "Worker Population", "Foragers Population", "Drones Population",
                "Total population"])
    plt.grid(True)
    plt.show()

    plot_without_total(values, x)


    