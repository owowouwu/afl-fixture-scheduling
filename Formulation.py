import pandas as pd
import numpy as np
from gurobipy import *

# Parameters
teams = ["Adelaide Crows", "Brisbane Lions", "Carlton Blues", "Collingwood Magpies",
     "Essendon Bombers", "Fremantle Dockers", "Geelong Cats", "Gold Coast Suns",
     "Greater Western Sydney Giants", "Hawthorn Hawks", "Melbourne Demons",
     "North Melbourne Kangaroos", "Port Adelaide Power", "Richmond Tigers",
     "St Kilda Saints", "Sydney Swans", "West Coast Eagles", "Western Bulldogs"]

team_numbers = {team: number for number, team in enumerate(sorted(teams), start=0)}
print(team_numbers['Carlton Blues'])

locations = ['VIC','NSW','SA','WA','QLD']
location_numbers = {location: number for number, location in enumerate(sorted(locations), start=0)}


home_locations = ['SA','QLD','VIC','VIC','VIC','WA','VIC','QLD','NSW','VIC',
                          'VIC','VIC','SA','VIC','VIC','NSW','WA','VIC']


ranking = [10,2,5,1,11,14,12,15,7,16,4,17,3,13,6,8,18,9] # 2023 pre-finals rankings

team_fans = [60,45,93,102,79,61,81,20,33,72,68,50,60,100,58,63,100,55] # 2023 number of members, in 000's

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


rivals = [['Port Adelaide Power'],['Gold Coast Suns','Collingwood Magpies'], # From wikipedia
          ['Essendon Bombers','Richmond Tigers','Collingwood Magpies', 'Fremantle Dockers'],
          ['Carlton Blues','Essendon Bombers','Brisbane Lions','Melbourne Demons','Richmond Tigers','Geelong Cats',
           'Hawthorn Hawks', 'West Coast Eagles','Port Adelaide Power'],
          ['Carlton Blues','Collingwood Magpies','Richmond Tigers','Hawthorn Hawks','North Melbourne Kangaroos'],
          ['Carlton Blues','West Coast Eagles'],['Collingwood Magpies','Hawthorn Hawks'],['Brisbane Lions'],
          ['Sydney Swans','Western Bulldogs'],['Hawthorn Hawks','Geelong Cats','Essendon Bombers', 'North Melbourne Kangaroos'],
          ['Collingwood Magpies'],['Essendon Bombers','Hawthorn Hawks'],['Adelaide Crows','Collingwood Magpies'],
          ['Carlton Blues','Collingwood Magpies','Essendon Bombers'],['Sydney Swans'],
          ['West Coast Eagles','St Kilda Saints','Greater Western Sydney Giants','Hawthorn Hawks'],
          ['Fremantle Dockers','Sydney Swans','Collingwood Magpies'],['Greater Western Sydney Giants']]

rivals_num = [[team_numbers[i] for i in rivals[j]] for j in range(len(rivals))]


timeslots = [i for i in range(7)]
timeslot_values = [100,130,50,60,110,50,40] # Change later according to attendances
timeslot_names = ['Thursday Night','Friday Night','Saturday Afternoon','Saturday Evening',
                  'Saturday Night','Sunday Afternoon','Sunday Evening']

rounds = [i for i in range(22)]

Ts = range(len(teams))
Ss = range(len(stadiums))
timeslots = range(7)
rounds = range(22)


# Decision Variables
fixture_matrix = [[[[[0 for i in Ts] for j in Ts] for s in Ss] for t in timeslots] for r in rounds]


# Attractiveness Function Parameters, adjust them as needed
alpha = 1.0
beta = 1.0
gamma = 1.0
sigma = 1.0
xi = 1.0

def attractiveness(i, j, s, t, r):
    score = 1
    
    if j in rivals[i]:
        score *= 1+alpha
    
    score /= max(abs(ranking[i]-ranking[j]),1)
    score  /= (ranking[i]+ranking[j])/2
    
    if stadium_locations[s] == home_locations[j]:
        score *= (1+beta)
        
    score *= stadium_size[s]
    score *= (team_fans[i]+team_fans[j])
    
    score *= timeslot_values[t]
    
    return score



def generate_initial_fixture():


    # Create a new model
    model = Model("fixture_scheduling")
    
    index = [(i, j, s, t, r) for i in Ts for j in Ts for s in Ss for t in timeslots for r in rounds]
    fixture = model.addVars(index,vtype=GRB.BINARY)
        
    for i in Ts: # Each team plays once a week
        for r in rounds:
            model.addConstr(quicksum(fixture[i, j, s, t, r] + fixture[j, i, s, t, r] for j in Ts for s in Ss for t in timeslots) == 1, "MatchesEachRound")
            
    
    for i in Ts:
        model.addConstr(quicksum(fixture[i, j, stadium_numbers[s], t, r] for j in Ts for s in home_stadiums[i] 
                                 for t in timeslots for r in rounds) == 11, f"HomeGames_{i}")
        
    for i in Ts:
        for s in list(set(stadiums)-set(home_stadiums[i])):
            model.addConstr(quicksum(fixture[i, j, stadium_numbers[s], t, r] for j in Ts for t in timeslots 
                                     for r in rounds) == 0, f"HomeGames_{i}")
    
    
    for i in Ts:
        model.addConstr(quicksum(fixture[i, i, s, t, r] for s in Ss for t in timeslots for r in rounds) <= 0, f"DoNotPlaySelf")
        for j in Ts:
            if i != j:
                model.addConstr(quicksum(fixture[i, j, s, t, r] + fixture[j, i, s, t, r] for s in Ss for t in timeslots for r in rounds) <= 2, f"PlayMoreThanTwice_{i}_{j}")
                model.addConstr(quicksum(fixture[i, j, s, t, r] + fixture[j, i, s, t, r] for s in Ss for t in timeslots for r in rounds) >= 1, f"PlayAtLeastOnce_{i}_{j}")
                
    #for i in Ts:
     #   for r in rounds[:-1]:
      #      model.addConstr(1 >= quicksum(fixture[i, j, s, t, r]+fixture[i, j, s, t, r] for j in Ts for s in Ss for t in [0,1]) + 
       #                     quicksum(fixture[i, j, s, t, r+1]+fixture[i, j, s, t, r+1] for j in Ts for s in Ss for t in [5,6]),
        #                    f"MoreThanFiveDayBreak_{i}")
            
    
    # Constraint 5: Two games in a row outside home location
    for i in Ts:
        for r in rounds[:-1]:
            model.addConstr(quicksum(fixture[j, i, stadium_numbers[s], t, r_] for j in Ts for s in list(set(stadiums) - set(home_location_stadiums[i])) 
                                 for t in timeslots for r_ in range(r,r+2)) <= 1, f"TwoGamesInRowOutside_{i}_{r}")
    
    # Constraint 6: 2+ Saturday games in the same stadium
    for r in rounds:
        for s in Ss:
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [2, 3, 4]) <= 1, f"SaturdayGamesInStadium_{r}_{s}")
    
    # Constraint 7: 2+ Sunday games in the same stadium
    for r in rounds:
        for s in Ss:
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [5, 6]) <= 1, f"SundayGamesInStadium_{r}_{s}")
    


    model.setObjective(quicksum(attractiveness(i,j,s,t,r)*fixture[i,j,s,t,r] for i in Ts for j in Ts 
                             for s in Ss for t in timeslots for r in rounds), GRB.MAXIMIZE)
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        for r in rounds:
            print('\n \n')
            print(f'Round {r}:')
            for t in timeslots:
                for i in Ts:
                    for j in Ts:
                        for s in Ss:
                            if fixture[i, j, s, t, r].x > 0.5:
                                print(f'{timeslot_names[t]}: {teams[i]} vs. {teams[j]} at {stadiums[s]}')
                                fixture_matrix[r][t][s][j][i] = 1
                                

    else:
        print("No feasible solution found.")
    
    
    return fixture_matrix

def feasibility(fixture):
    violated = 0
    critical = 0
    
    if sum(fixture[i, j, s, t, r] for i in Ts for j in Ts for s in Ss for t in timeslots for r in rounds) != 9*22:
        critical += 1
        violated += 1 # Number of total matches
        
    for i in Ts:
        if sum(fixture[i, j, s, t, r] for j in teams for s in stadiums for t in timeslots for r in rounds) != 11: # 11 home games
            violated += 1 # Number of home games
        
        violated += sum(fixture[i, j, stadium_numbers[s], t, r] for j in Ts for s in list(set(stadiums).difference(home_stadiums[i])) 
                                for t in timeslots for r in rounds) # Home games outside home ground
        
        for j in Ts:
            vs = sum(fixture[i, j, s, t, r] for s in Ss for t in timeslots for r in rounds)
            if i == j:
                violated += vs # Can't play yourself
                critical += vs
            
            else:
                if vs == 0:
                    violated += 1 # Don't play the other team
                elif vs > 2:
                    violated+= vs-1 # Play the other team too much
        
        last = 0
        for r in rounds:
            current = sum(fixture[i, j, stadium_numbers[s], t, r] for j in Ts for s in list(set(stadiums).difference(home_location_stadiums[i])) for t in timeslots)
            violated += last*current # Two games in a row outside home location
            last = current

            
    for r in rounds:
        for s in Ss:
            if sum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [2,3,4]) > 1: # 2+ Saturday games in stadium
                violated += 1
                critical += 1
                
            if sum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [5,6]) > 1: # 2+ Sunday games in stadium
                violated += 1
                critical += 1
            
    return violated, critical



def fixture_attractiveness(fixture,max_value,violated_factor,critical_factor):
    total_score = 0
    
    for r in rounds:
        for t in timeslots:
            value = 0
            for i in teams:
                for j in teams:
                    for s in stadiums:
                        if fixture[i,j,s,t,r] == 1:
                            value += attractiveness(i, j, s, t, r)
                            
            total_score += min(max_value,value)
            
    violated, critical = feasibility(fixture)
    return total_score - violated_factor*violated - critical_factor*critical
    

fixture_matrix = generate_initial_fixture()
np.save('solutions/mip_initial_fixture.npy', fixture_matrix)
