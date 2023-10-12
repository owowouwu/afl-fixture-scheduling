import pandas as pd
import numpy as np
from gurobipy import *
import random
import itertools
import time 
import pickle as pkl 


# Parameters
teams = ["Adelaide Crows", 
         "Brisbane Lions", 
         "Carlton Blues", 
         "Collingwood Magpies",
         "Essendon Bombers", 
         "Fremantle Dockers", 
         "Geelong Cats", 
         "Gold Coast Suns",
         "Greater Western Sydney Giants", 
         "Hawthorn Hawks", 
         "Melbourne Demons",
         "North Melbourne Kangaroos", 
         "Port Adelaide Power",
         "Richmond Tigers",
         "St Kilda Saints",
         "Sydney Swans",
         "West Coast Eagles",
         "Western Bulldogs"]

team_numbers = {team: number for number, team in enumerate(sorted(teams), start=0)}

locations = ['VIC','NSW','SA','WA','QLD']

location_numbers = {location: number for number, location in enumerate(sorted(locations), start=0)}

home_locations = ['SA','QLD','VIC','VIC','VIC','WA','VIC','QLD','NSW','VIC',
                          'VIC','VIC','SA','VIC','VIC','NSW','WA','VIC']

ranking = [10,2,5,1,11,14,12,15,7,16,4,17,3,13,6,8,18,9] # 2023 pre-finals rankings

team_fans = [60,45,93,102,79,61,81,20,33,72,68,50,60,100,58,63,100,55] # 2023 number of members, in 000's

wins = [11,17,13.5,18,11,10,10.5,9,13,7,16,3,17,10.5,13,12.5,3,12]

stadiums = ['MCG','Marvel','GMHBA','Adelaide Oval','Optus','Gabba','HBS','SCG','Giants']

stadium_numbers = {stadium: number for number, stadium in enumerate(sorted(stadiums), start=0)}

stadium_locations = ['VIC','VIC','VIC','SA','WA','QLD','QLD','NSW','NSW']

home_stadiums = [['Adelaide Oval'],['Gabba'],['MCG','Marvel'],['MCG','Marvel'],['MCG','Marvel'],['Optus'],['GMHBA'],
                 ['HBS'],['Giants'],['MCG'],['MCG'],['Marvel'],['Adelaide Oval'],
                 ['MCG'],['Marvel'],['SCG'],['Optus'],['Marvel']]

home_location_stadiums = [[] for i in range(len(teams))]
for i in range(len(teams)):
    for j in range(len(stadiums)):
        if stadium_locations[j] == home_locations[i]:
            home_location_stadiums[i].append(stadiums[j])

stadium_size = [100,53,40,54,60,39,27,47,24] # Stadium sizes in 000's

rivals = [['Port Adelaide Power'],
          ['Gold Coast Suns','Collingwood Magpies'], # From wikipedia
          ['Essendon Bombers','Richmond Tigers','Collingwood Magpies', 'Fremantle Dockers'],
          ['Carlton Blues','Essendon Bombers','Brisbane Lions','Melbourne Demons','Richmond Tigers','Geelong Cats',
           'Hawthorn Hawks', 'West Coast Eagles','Port Adelaide Power'],
          ['Carlton Blues','Collingwood Magpies','Richmond Tigers','Hawthorn Hawks','North Melbourne Kangaroos'],
          ['Carlton Blues','West Coast Eagles'],
          ['Collingwood Magpies','Hawthorn Hawks'],
          ['Brisbane Lions'],
          ['Sydney Swans','Western Bulldogs'],
          ['Hawthorn Hawks','Geelong Cats','Essendon Bombers', 'North Melbourne Kangaroos'],
          ['Collingwood Magpies'],
          ['Essendon Bombers','Hawthorn Hawks'],
          ['Adelaide Crows','Collingwood Magpies'],
          ['Carlton Blues','Collingwood Magpies','Essendon Bombers'],
          ['Sydney Swans'],
          ['West Coast Eagles','St Kilda Saints','Greater Western Sydney Giants','Hawthorn Hawks'],
          ['Fremantle Dockers','Sydney Swans','Collingwood Magpies'],
          ['Greater Western Sydney Giants']]

rivals_num = [[team_numbers[i] for i in rivals[j]] for j in range(len(rivals))]

timeslots = [i for i in range(7)]
timeslot_values = [10,13,5,6,11,5,4] # Change later according to attendances
timeslot_names = ['Thursday Night','Friday Night','Saturday Afternoon','Saturday Evening',
                  'Saturday Night','Sunday Afternoon','Sunday Evening']

Ts = range(len(teams))
Ss = range(len(stadiums))
timeslots = range(7)
rounds = range(22)

# Decision Variables
fixture_matrix = [[[[[0 for r in rounds] for t in timeslots] for s in Ss] for j in Ts] for i in Ts]

# Attractiveness Function Parameters, adjust them as needed
alpha = 1.0
beta = 1.0
gamma = 1.0
sigma = 1.0
xi = 1.0

def attractiveness(i, j, s, t, r):
    score = 1
    if r == 0:
        score *= 4
    elif r == 1:
        score *= 2
    elif r == 21:
        score *= 2
    
    if j in rivals[i]:
        score *= 1+alpha
    
    score /= np.sqrt(1+abs(ranking[i]-ranking[j]))
    score  /= np.sqrt(ranking[i]+ranking[j])
    
    if stadium_locations[s] == home_locations[j]:
        score *= (1+beta)
        
    score *= np.sqrt(stadium_size[s])
    score *= np.sqrt(team_fans[i]+0.5*team_fans[j])
    
    score *= timeslot_values[t]
    
    return score


def probability_win(i, j, s):
    probability = wins[i]/(wins[i]+wins[j])
    if stadiums[s] not in home_location_stadiums[j]:
        probability += (1-probability)/2.5
    elif stadiums[s] not in home_stadiums[j]:
        probability += (1-probability)/4
    else:
        probability += (1-probability)/10
        
    return probability

def expected_win_variance(fixture):
    results = []
    expected_wins = [0]*18
    for r in rounds:
        for i in Ts:
            for j in Ts:
                for s in Ss:
                    for t in timeslots:
                        expected_wins[i] += probability_win(i, j, s)*fixture[i][j][s][t][r]
                        expected_wins[i] += (1-probability_win(j, i, s))*fixture[j][i][s][t][r]
        
        results.append(np.var(expected_wins))

    return sum((i+1)*results[i] for i in range(len(results)))
    
def feasibility_print(fixture):
    violated = 0
    critical = 0
    
    for i in Ts: # Each team plays once a week
        for r in rounds:
            critical += abs(sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for j in Ts for s in Ss for t in timeslots)-1)
            print('Violated Constraint 1')
    
    for i in Ts:
        critical += abs(sum(fixture[i][j][stadium_numbers[s]][t][r] for j in Ts for s in home_stadiums[i] 
                                 for t in timeslots for r in rounds)-11)
        print('Violated Constraint 2')
    
    
    for i in Ts:
        critical += sum(fixture[i, i, s, t, r] for s in Ss for t in timeslots for r in rounds)
        print('Violated Constraint 3')
        
        for j in Ts:
            if i != j:
                violated += max(sum(fixture[i][j][s][t][r] for s in Ss for t in timeslots for r in rounds)-1,0)
                violated += max(1-sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for s in Ss for t in timeslots for r in rounds),0)
                print('Violated Constraint 4')
                
                
    for i in Ts:
        for r in rounds[:-1]:
            violated += max(0,sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for j in Ts for s in Ss for t in [5,6])+ 
                            sum(fixture[i, j, s, t, r+1]+fixture[j, i, s, t, r+1] for j in Ts for s in Ss for t in [0]) - 1)
            print('Violated Constraint 5')
            
    
    # Three games in a row outside home location
    for i in Ts:
        for r in rounds[:-2]:
            violated += max(0,1-sum(fixture[j][i][stadium_numbers[s]][t][r_]+fixture[i][j][stadium_numbers[s]][t][r_] for j in Ts for s in home_location_stadiums[i] 
                                 for t in timeslots for r_ in range(r,r+3)))
            print('Violated Constraint 6')
       
    # Four away games in a row
    for i in Ts:
        for r in rounds[:-3]:
            violated += max(0,1-sum(fixture[i, j, s, t, r_] for j in Ts for s in Ss for t in timeslots for r_ in range(r,r+4)))
            print('Violated Constraint 7')
    
    # Constraint 7: 2+ games in one day in the same stadium
    for r in rounds:
        for s in Ss:
            
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for t in [5, 6])-1)
            print('Violated Constraint 8')
            
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for t in [2, 3, 4])-1)
            print('Violated Constraint 9')
            
            for t in [0,1]:
                violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts)-1)
                print('Violated Constraint 10')
    
    
    # Constraint: No more than two games in any timeslot, and only one on Thursday and Friday night, at least one in each
    for r in rounds:
        
        for t in [2,3,4,5,6]:
            violated += max(0,1-sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)) # At least one game each timeslot
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)-2)
            print('Violated Constraint 11')
        
        for t in [0,1]:
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)-1) # One game
            print('Violated Constraint 12')
            
            
    return violated, critical

def feasibility(fixture):
    violated = 0
    critical = 0
    
    for i in Ts: # Each team plays once a week
        for r in rounds:
            critical += abs(sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for j in Ts for s in Ss for t in timeslots)-1)
            
    
    for i in Ts:
        critical += abs(sum(fixture[i][j][stadium_numbers[s]][t][r] for j in Ts for s in home_stadiums[i] 
                                 for t in timeslots for r in rounds)-11)
    
    
    for i in Ts:
        critical += sum(fixture[i, i, s, t, r] for s in Ss for t in timeslots for r in rounds)
        
        for j in Ts:
            if i != j:
                violated += max(sum(fixture[i][j][s][t][r] for s in Ss for t in timeslots for r in rounds)-1,0)
                violated += max(1-sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for s in Ss for t in timeslots for r in rounds),0)
                
                
    for i in Ts:
        for r in rounds[:-1]:
            violated += max(0,sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for j in Ts for s in Ss for t in [5,6])+ 
                            sum(fixture[i, j, s, t, r+1]+fixture[j, i, s, t, r+1] for j in Ts for s in Ss for t in [0]) - 1)
            
    
    # Three games in a row outside home location
    for i in Ts:
        for r in rounds[:-2]:
            violated += max(0,1-sum(fixture[j][i][stadium_numbers[s]][t][r_]+fixture[i][j][stadium_numbers[s]][t][r_] for j in Ts for s in home_location_stadiums[i] 
                                 for t in timeslots for r_ in range(r,r+3)))
       
    # Four away games in a row
    for i in Ts:
        for r in rounds[:-3]:
            violated += max(0,1-sum(fixture[i, j, s, t, r_] for j in Ts for s in Ss for t in timeslots for r_ in range(r,r+4)))
    
    
    # Constraint 7: 2+ games in one day in the same stadium
    for r in rounds:
        for s in Ss:
            
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for t in [5, 6])-1)
            
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for t in [2, 3, 4])-1)
            
            for t in [0,1]:
                violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts)-1)
    
    
    # Constraint: No more than two games in any timeslot, and only one on Thursday and Friday night, at least one in each
    for r in rounds:
        
        for t in [2,3,4,5,6]:
            violated += max(0,1-sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)) # At least one game each timeslot
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)-2)
        
        for t in [0,1]:
            violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)-1) # One game
            
            
    return violated, critical



def fixture_attractiveness(fixture,max_value,violated_factor,critical_factor,equality_factor):
    total_score = 0
    
    for r in rounds:
        for t in timeslots:
            value = 0
            for i in Ts:
                for j in Ts:
                    for s in Ss:
                        value += attractiveness(i, j, s, t, r)*fixture[i][j][s][t][r]
                            
            total_score += min(max_value,value)
            
    violated, critical = feasibility(fixture)
    equality = equality_factor*expected_win_variance(fixture)
    
    return total_score - violated_factor*violated - critical_factor*critical - equality

def objective_value(fixture):

    violated, critical = feasibility(fixture)

    violated_penalty = 2*10^4
    critical_penalty = 10^6

    objective_value = - violated_penalty*violated - critical_penalty*critical + fixture_attractiveness(fixture, 2*10^4, critical_penalty, violated_penalty, 10^3) 

    return objective_value

def genesis(initial_sol_type):
    print('Genesis')

    if initial_sol_type == 'MILP':
        # MIP SOLUTION
        with open('solutions/MILP_Fixture_1000.pkl', 'rb') as pkl_file:
            MIP_soln1 = pkl.load(pkl_file)
        with open('solutions/MILP_Fixture_10000.pkl', 'rb') as pkl_file:
            MIP_soln2 = pkl.load(pkl_file)
        with open('solutions/MILP_Fixture_100000.pkl', 'rb') as pkl_file:
            MIP_soln3 = pkl.load(pkl_file)
        with open('solutions/MILP_Fixture_1000000.pkl', 'rb') as pkl_file:
            MIP_soln4 = pkl.load(pkl_file)
        
        solns = [MIP_soln1, MIP_soln2, MIP_soln3, MIP_soln4]

    elif initial_sol_type == 'greedy':
        solns = [np.load(f'ga_input/greedy{i}.npy') for i in range(1,5)]
    
    pop = []
    for soln in solns:
        pop.append([soln, objective_value(soln)])
    
    return pop

import random

def select_parents(population):
    """
    Select two individuals from a population for reproduction.
    
    Args:
    - population (list): A list of individuals, where each individual is represented as a tuple (individual_data, objective_value).
    
    Returns:
    - parent1 (tuple): The first parent selected.
    - parent2 (tuple): The second parent selected.
    """
    
    print('Romance')
    # Sort the population by objective value in descending order (higher is better)
    sorted_population = sorted(population, key=lambda x: x[1], reverse=True)

    # Calculate the number of elite individuals (top 25%)
    num_elite = len(sorted_population) // 4

    if num_elite > 1:
        # If there are enough elite individuals, select the first parent as an elite individual
        parent1 = random.choice(sorted_population[:num_elite])
    else:
        # If there are not enough elite individuals, choose both parents randomly
        parent1 = random.choice(sorted_population)
    
    # Select the second parent randomly from the entire population
    parent2 = random.choice(population)

    return parent1, parent2

def birth(parent1, parent2):
    print('Birth')
    # print(get_dimensions(parent1), get_dimensions(parent2))
    # Determine a random crossover point
    crossover_point = np.random.randint(0, len(parent1))  # Include 0, exclude len(parent1)

    # Create the first child by combining the first part of parent1 and the second part of parent2
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

    # Create the second child by combining the first part of parent2 and the second part of parent1
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return [child1, objective_value(child1)], [child2, objective_value(child2)]

def evolutionPlus():

    start_time = time.time()  # Record the start time

    # Parameters 
    solns = []
    num_gen = 0
    duration = 60 * 10

    pop = genesis()  # Use the provided pop_size when calling genesis

    best_soln = max(pop, key=lambda x: x[1])
    while time.time() - start_time < duration:
        num_gen += 1
        # Select Parents
        parent1, parent2 = select_parents(pop)

        # Create Children
        child1, child2 = birth(parent1[0], parent2[0])
        pop.append(child1)
        pop.append(child2)
        
        # Death
        family = [parent1, parent2, child1, child2]
        # Find family member with the minimum objective value
        min_value = 1000000000000
        to_die_idx = None
        for i, member in enumerate(family):
            if member[1] < min_value:
                min_value = member[1]
                to_die_idx = i

        # Remove the element at to_die_idx
        if to_die_idx is not None:
            print('Death')
            pop.pop(to_die_idx)

        # Find the index of the element with the largest objective value
        max_index = 0
        max_value = pop[0][1]
        for i, soln in enumerate(pop):
            if soln[1] > max_value:
                max_value = soln[1]
                max_index = i

        # Extract the element with the largest objective value 
        best_soln = pop[max_index]
        solns.append(best_soln)

    for i, soln in enumerate(pop):
        if soln[1] > max_value:
            max_index = i
    best_soln = pop[max_index]

    print('Solution: ', best_soln[0])
    print('Value: ', best_soln[1])

    return best_soln

best_soln = evolutionPlus()
print(best_soln)

violated, critical = feasibility(best_soln[0])
print("Number of Violated Critical Constraints: ",critical)
print("Number of Violated Constraints: ",violated)

import sys

with open("output/GA_greedy_output_raw.txt", "w") as file:
    sys.stdout = file
    print(best_soln)  # This will be written to output.txt

with open("output/GA_greedy_output_interpreted.txt", "w") as file:
    sys.stdout = file 
    best_fixture = best_soln[0]
    for r in rounds: 
        print('\nRound ', r)
        for t in timeslots:
            for i in Ts:
                for j in Ts:
                    for s in range(len(Ss)-1):
                        if best_fixture[i][j][s][t][r] == 1:
                            if j in [team_numbers[rival] for rival in rivals[i]]:
                                print("Rivalry Match! ", teams[i], " VS ", teams[j], ' AT ', stadiums[s], ' ON ', timeslot_names[t])
                            else: 
                                print(teams[i], " VS ", teams[j], ' AT ', stadiums[s], ' ON ', timeslot_names[t])
                            