Established in 1897, The Australian Football League (AFL) is one of the most popular Australian Sports Leagues, attracting over 125 million spectators in 2022. Generating the annual AFL fixture is a difficult task, requiring a balance between many, often conflicting, interests. For example, competition fairness, revenue, spectator expectations, team expectations, and resource availability, among other factors, must be considered. An interconnected web of objectives and constraints is navigated to build an AFL fixture that balances multiple objectives and adheres to the unique characteristics of the league. \\

In this project we aim to implement and combine a number of heuristic algorithms and a mixed integer programming formulation of the problem. For details, see `report.pdf`.

# Requirements

- Gurobi (MILP Solution)
- NumPy
- Pandas

# Usage

The root directory contains a number of scripts for all the algorithms that have been implemented. 

- `Formulation.py` contains a mixed integer programming solution for our problem using Gurobi as a black box solver. `Formulation_Genesis.py` is used to solve relaxations of the problem to generate initial populations for our genetic algorithm.
- `random_greedy_fixtures1.py` and `random_greedy_fixtures2.py` use two greedy heuristics.
- `grasp.py` contains implements GRASP using the aforementioned greedy heuristics.
- `GA_base.py`, `GA_improved.py` are our implementations of the genetic algorithm. The improved version is elaborated upon further in the report.
- `simulated_annealing.ipynb` contains code for simulated annealing.

A couple bash scripts have also been provided in order to run the greedy and GRASP heuristics for multiple seeds.


