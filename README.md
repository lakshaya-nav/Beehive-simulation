# To bee or not to bee?
## Simulating bee colony population dynamics

This project aims to investigate how internal population dynamics and the spread of Queen Mandibular Pheromone (QMP), influence both the total colony population and bee caste dynamics. The impact of different climates is also explored through a case study. In particular, the model illustrates the role of overcrowding and pheromone distribution in triggering swarming events and the recursive formation of child beehives. By integrating time-based population and spatial models with stochastic parameterisation, the project captures both the macroscopic population trends and microscopic bee interactions that determine colony-level decision-making, resulting in a biologically accurate model.
 
To run the main population simulation (which includes the pheromone spreading simulation) the mainCode.py file should be run. The simulation parameters can be adjusted in the main funtction where simulate is called. 
The simulate function is called with parameters as follows: 
simulate(simulation_horizon, population_simulation_step=1, initial_state_of_beehive, colony_parameters, pheromone_simulation_run_step, temperatures=None)

To run the climate case study the fille caseStudy.py should be run. The .csv files containing the temperatures need to be in the same folder as the caseStudy.py and mainCode.py files. The .csv files for the experimentation in the report are provided. To adjust the parameters of each climate, the simulate() calls in each funtion corrisponding to a country should be adjusted.

To run the recursive child beehive formation simulation, the file recursive_beehive_formation.py() should be run. 

This simulation utilizes many python libraries and toolboxes.
