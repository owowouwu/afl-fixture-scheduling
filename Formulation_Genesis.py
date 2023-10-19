import pandas as pd
import numpy as np
import gurobipy as GRB 
from gurobipy import *
import pickle

# Parameters
teams = ["Adelaide Crows", "Brisbane Lions", "Carlton Blues", "Collingwood Magpies",
     "Essendon Bombers", "Fremantle Dockers", "Geelong Cats", "Gold Coast Suns",
     "Greater Western Sydney Giants", "Hawthorn Hawks", "Melbourne Demons",
     "North Melbourne Kangaroos", "Port Adelaide Power", "Richmond Tigers",
     "St Kilda Saints", "Sydney Swans", "West Coast Eagles", "Western Bulldogs"]

team_numbers = {team: number for number, team in enumerate(teams, start=0)}

locations = ['Victoria','NSW','SA','WA','QLD']
location_numbers = {location: number for number, location in enumerate(locations, start=0)}


home_locations = ['SA','QLD','Victoria','Victoria','Victoria','WA','Victoria','QLD','NSW','Victoria',
                          'Victoria','Victoria','SA','Victoria','Victoria','NSW','WA','Victoria']
#2022 Stats

ranking = [14,9,6,4,15,5,1,12,16,13,2,18,11,7,10,3,17,8] # 2022 post-finals rankings

wins = [8,15,12,16,7,15,18,10,6,8,16,2,10,13,11,16,2,12] # 2022 pre-finals wins

team_fans = [63,43,89,100,86,56,72,21,30,81,66,50,59,101,60,55,103,51] # 2022 number of members, in 000's

#2023 Stats 

# ranking = [10,2,5,1,11,14,12,15,7,16,4,17,3,13,6,8,18,9] # 2023 pre-finals rankings

# wins = [11,17,13.5,18,11,10,10.5,9,13,7,16,3,17,10.5,13,12.5,3,12]

# team_fans = [60,45,93,102,79,61,81,20,33,72,68,50,60,100,58,63,100,55] # 2023 number of members, in 000's

stadiums = ['MCG','Marvel','GMHBA','Adelaide Oval','Optus','Gabba','HBS','SCG','Giants']
stadium_numbers = {stadium: number for number, stadium in enumerate(stadiums, start=0)}

stadium_locations = ['Victoria','Victoria','Victoria','SA','WA','QLD','QLD','NSW','NSW']

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
timeslot_values = [10,13,6,7,11,5,4] # Change later according to attendances
timeslot_names = ['Thursday Night','Friday Night','Saturday Afternoon','Saturday Evening',
                  'Saturday Night','Sunday Afternoon','Sunday Evening']

rounds = [i for i in range(22)]

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
    
    print(results)
    return sum((i+1)*results[i] for i in range(len(results)))
    


def generate_initial_fixture():


    # Create a new model
    model = Model("fixture_scheduling")
    model.setParam('Timelimit', 7200)
    model.setParam("MIPGap", 0.01)
    
    index = [(i, j, s, t, r) for i in Ts for j in Ts for s in Ss for t in timeslots for r in rounds]
    timeslots_index = [(t, r) for t in timeslots for r in rounds]



    fixture = model.addVars(index,vtype=GRB.BINARY)
    #game_on = model.addVars(timeslots_index,vtype=GRB.BINARY)
    equality = model.addVars([(i) for i in Ts])
        
    
    for i in Ts: # Each team plays once a week
        for r in rounds:
            model.addConstr(quicksum(fixture[i, j, s, t, r] + fixture[j, i, s, t, r] for j in Ts for s in Ss for t in timeslots) == 1, "MatchesEachRound")
            
    
    for i in Ts: # 11 Home Games 
        model.addConstr(quicksum(fixture[i, j, stadium_numbers[s], t, r] for j in Ts for s in home_stadiums[i] 
                                 for t in timeslots for r in rounds) == 11, f"HomeGames_{i}")
    
    
    for i in Ts:
        model.addConstr(quicksum(fixture[i, i, s, t, r] for s in Ss for t in timeslots for r in rounds) <= 0, f"DoNotPlaySelf")
        for j in Ts:
            if i != j:
                model.addConstr(quicksum(fixture[i, j, s, t, r] for s in Ss for t in timeslots for r in rounds) <= 1, f"PlayMoreThanTwice_{i}_{j}")
                model.addConstr(quicksum(fixture[i, j, s, t, r] + fixture[j, i, s, t, r] for s in Ss for t in timeslots for r in rounds) >= 1, f"PlayAtLeastOnce_{i}_{j}")
                
                
    for i in Ts:
        for r in rounds[:-1]:
            model.addConstr(1 >= quicksum(fixture[i, j, s, t, r]+fixture[j, i, s, t, r] for j in Ts for s in Ss for t in [5,6]) + 
                            quicksum(fixture[i, j, s, t, r+1]+fixture[j, i, s, t, r+1] for j in Ts for s in Ss for t in [0]),
                           f"AtLeastAFiveDayBreak_{i}")
            
    
    # Three games in a row outside home location
    for i in Ts:
        for r in rounds[:-2]:
            model.addConstr(quicksum(fixture[j, i, stadium_numbers[s], t, r_]+fixture[i,j, stadium_numbers[s], t, r_] for j in Ts for s in home_location_stadiums[i] 
                                 for t in timeslots for r_ in range(r,r+3)) >= 1, f"ThreeGamesInRowOutside_{i}_{r}")
       
    # Four away games in a row
    for i in Ts:
        for r in rounds[:-3]:
            model.addConstr(quicksum(fixture[i, j, s, t, r_] for j in Ts for s in Ss for t in timeslots
                                     for r_ in range(r,r+4)) >= 1, f"FourAwayGamesInRow_{i}_{r}")
    
    
    # Constraint 7: 2+ games in one day in the same stadium
    for r in rounds:
        for s in Ss:
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [5, 6]) <= 1, f"SundayGamesInStadium_{r}_{s}")
            
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for t in [2, 3, 4]) <= 1, f"SaturdayGamesInStadium_{r}_{s}")
            
            for t in [0,1]:
                model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts) <= 1, f"OneGameAtATimeInStadium_{r}_{s}")
    
    
    # Constraint: No more than two games in any timeslot, and only one on Thursday and Friday night
    for r in rounds:
        
        model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for s in Ss for t in [5,6]) >= 2, f"AtLeastTwoSundayGames")
        
        for t in [0,1]:
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for s in Ss) <= 1, f"OneThursday&FridayNightGame_{r}")
            
        for t in [2,3,4,5,6]:
            model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for s in Ss) <= 2, f"NoMoreThanTwoSimultaneousGames_{r}")
        
        #for t in timeslots:
         #   model.addConstr(quicksum(fixture[i, j, s, t, r] for i in Ts for j in Ts for s in Ss) >= game_on[t,r], f"IncentiviseAtLeastOneGameInEachTimeslot_{r}_{t}")
            
    
    # These find the absolute value for how many wins away from the AFL average each team is (the average numbers of wins in a 22 game season is 11).    
    for i in Ts:
        model.addConstr(equality[i] >= quicksum(probability_win(i, j, s)*fixture[i,j,s,t,r] + (1-probability_win(j,i,s))*fixture[j,i,s,t,r] 
                                                   for j in Ts for s in Ss for t in timeslots for r in rounds)-11)
                
        model.addConstr(equality[i] >= 11-quicksum(probability_win(i, j, s)*fixture[i,j,s,t,r] + (1-probability_win(j,i,s))*fixture[j,i,s,t,r] 
                                                      for j in Ts for s in Ss for t in timeslots for r in rounds))

    # We remove the requirement of at least one game in each timeslot, it leads to errors
    model.setObjective(quicksum(attractiveness(i,j,s,t,r)*fixture[i,j,s,t,r] for i in Ts for j in Ts for s in Ss for t in timeslots for r in rounds) 
                       - 100*quicksum(equality[i] for i in Ts), GRB.MAXIMIZE)
    #+ 250*quicksum(game_on[t,r] for t in timeslots for r in rounds) 
    
    model.optimize()
    
    
    if model.status == GRB.OPTIMAL:
        for r in rounds:
            print('\n \n')
            print(f'Round {r+1}:')
            for t in timeslots:
                for i in Ts:
                    for j in Ts:
                        for s in Ss:
                            if fixture[i, j, s, t, r].x > 0.5:
                                print(f'{timeslot_names[t]}: {teams[i]} vs. {teams[j]} at {stadiums[s]}')
                                fixture_matrix[i][j][s][t][r] = 1                         

    else:
        print("No feasible solution found.")
    
    
    return np.array(fixture_matrix), model.objVal



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
        
        #for t in [0,1]:
         #   violated += max(0,sum(fixture[i][j][s][t][r] for i in Ts for j in Ts for s in Ss)-1) # One game
            
            
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

def main():
  
    MILP_fixture, MILP_value = generate_initial_fixture()
    print(MILP_value)  

    max_value,violated_factor,critical_factor,equality_factor = 2*(10**4),2*10**4,10**6,10**3
    value = fixture_attractiveness(MILP_fixture,max_value,violated_factor,critical_factor,equality_factor)

    print(value)
    path = 'initialPopulation'
    with open(f'{path}\MILP_Fixture_2022.pkl', 'wb') as file:
        pickle.dump(MILP_fixture, file)

if __name__ == '__main__':
    main()

  
# MILP_fixture, MILP_value = generate_initial_fixture()
# print(MILP_value)

max_value,violated_factor,critical_factor,equality_factor = 2*(10**4),2*10**4,10**6,10**3
MILP_fixture, MILP_value = generate_initial_fixture()
value = fixture_attractiveness(MILP_fixture,max_value,violated_factor,critical_factor,equality_factor)
print(value)
print(MILP_fixture)
np.save('MILP_Fixture_100_2022.npy', MILP_fixture)
