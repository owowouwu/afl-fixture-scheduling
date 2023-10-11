import numpy as np
import json
from tournament import Tournament
from modules.greedy_round_scheduler import GraspHeuristic

np.random.seed(1234)

with open('teams.json', 'r') as f:
    teams = json.load(f)

with open('locations.json', 'r') as f:
    locations = json.load(f)

stadiums = {}
for loc in locations:
    snames = []
    for s in locations[loc]['stadiums']:
        snames.append(s)
        stadiums[s] = {
            'location': loc,
            'size': locations[loc]['stadiums'][s]['size'] 
        }
    
    locations[loc]['stadiums'] = snames

timeslot_values = [100,130,40,50,80,110,50,40,90] # Change later according to attendances
timeslot_names = ['Thursday Night','Friday Night','Saturday Afternoon','Saturday Evening',
                  'Saturday Night','Sunday Afternoon','Sunday Evening']
timeslots = [{'value': v, 'name': n} for (n,v) in zip(timeslot_names, timeslot_values)]
tourn = Tournament(teams = teams, locations = locations, stadiums=stadiums, timeslots = timeslots, rounds = 22)

gr = GraspHeuristic(tourn)
arr = tourn.weight_matrix
fixture, obj = gr.grasp_heuristic(iterations = 10, local_it = 10, rcl_length = 10,
                                    attractiveness_matrix=arr, rounds = 22, timeslots = 9)
tourn.fixture_matrix = fixture
tourn.print_fixture()
print("objective: ", tourn.fixture_attractiveness(fixture=fixture))
print("constraints violated: ", tourn.feasibility(fixture, debug=True))