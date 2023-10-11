import numpy as np
from datetime import datetime
import pandas as  pd
import argparse
import sys 
import os

from Formulation import probability_win, attractiveness


teams = ["Adelaide Crows", "Brisbane Lions", "Carlton", "Collingwood",
     "Essendon", "Fremantle", "Geelong Cats", "Gold Coast Suns",
     "GWS Giants", "Hawthorn", "Melbourne",
     "North Melbourne", "Port Adelaide", "Richmond",
     "St Kilda", "Sydney Swans", "West Coast Eagles", "Western Bulldogs"]

team_numbers = {team: number for number, team in enumerate(teams, start=0)}

stadiums = ['MCG','Marvel Stadium','GMHBA Stadium','Adelaide Oval','Optus Stadium','Gabba',
            'Heritage Bank Stadium','SCG','GIANTS Stadium']
stadium_numbers = {stadium: number for number, stadium in enumerate(stadiums, start=0)}
home_stadiums = [['Adelaide Oval'],['Gabba'],['MCG','Marvel Stadium'],['MCG','Marvel Stadium'],['MCG','Marvel Stadium'],['Optus'],['GMHBA Stadium'],
                 ['Heritage Bank Stadium'],['GIANTS Stadium'],['MCG'],['MCG'],['Marvel Stadium'],['Adelaide Oval'],
                 ['MCG'],['Marvel Stadium'],['SCG'],['Optus Stadium'],['Marvel Stadium']]

# add 


timeslots = [i for i in range(7)]
timeslot_names = ['Thursday Night','Friday Night','Saturday Afternoon','Saturday Evening',
                  'Saturday Night','Sunday Afternoon','Sunday Evening']

rounds = [i for i in range(22)]

Ts = range(len(teams))
Ss = range(len(stadiums))
timeslots = range(7)
rounds = range(22)



def fixture_equality(fixture):
    inequality = 0
    for i in Ts:
        wins = 0
        games = 0
        for r in rounds:
            for j in Ts:
                for s in Ss:
                    for t in timeslots:
                        wins += probability_win(i, j, s)*fixture[i][j][s][t][r] + (1-probability_win(j, i, s))*fixture[j][i][s][t][r]
                        games += fixture[i][j][s][t][r] + fixture[j][i][s][t][r]
                        
            
            inequality += games*(wins-games/2)**2 / 18
            
    return inequality

def fixture_attractiveness(fixture,equality_factor,max_value):
    total_score = 0
    
    for r in rounds:
        for t in timeslots:
            value = 0
            for i in Ts:
                for j in Ts:
                    for s in Ss:
                        value += attractiveness(i, j, s, t, r)*fixture[i][j][s][t][r]
                            
            total_score += min(max_value,value)
            
    equality = equality_factor*fixture_equality(fixture)
    
    return total_score - equality



def csv_to_fixture(data):


    # Decision Variables
    fixture_matrix = [[[[[0 for r in rounds] for t in timeslots] for s in Ss] for j in Ts] for i in Ts]
    
    for _,row in data.iterrows():
        try:
            i,j,s,t,r = row["Home Team"],row["Away Team"],row["Location"],row["Date"],int(row["Round Number"])
            
        except Exception: # A post-regular season match
            continue
        
        if r < 5:
            r -= 1
        elif r > 14: # After bye
            r -= 3
        elif r > 5:
            r -= 2        
        else: # r = 5, i.e. Gather Round
            continue
    
        # We ignore random stadiums across Australia (these have small capacity anyway)
        if i == "Gold Coast Suns":
            s = "Heritage Bank Stadium"
        elif i in ["Hawthorn","North Melbourne","Western Bulldogs","Melbourne"]: # Convert out of state stadiums to Marvel
            if s != "MCG":
                s = "Marvel Stadium"
        elif i == "GWS Giants":
            s = "GIANTS Stadium"

        # if none of these above checks work then we assign stadiums appropriate to the home team
        if s not in stadium_numbers:
            s = home_stadiums[team_numbers[i]][0]
            
            
        i,j = team_numbers[i],team_numbers[j]
        s = stadium_numbers[s]
        
        t_object = datetime.strptime(t,"%d/%m/%Y  %H:%M") # 13/05/2023  1:45:00 PM
        
        if t_object.weekday() == 3:
            t = 0
        elif t_object.weekday() == 4:
            t = 1
            
        elif t_object.weekday() == 6: # Sunday
            if t_object.hour < 16:
                t = 5
            else:
                t = 6
            
        else: # Saturday (or primetime on weekdays)
            if t_object.hour < 16:
                t = 2
            elif t_object.hour < 19:
                t = 3
            else:
                t = 4
        
        fixture_matrix[i][j][s][t][r] = 1
        
    return fixture_matrix



def main(data_file):
    # Your code here
    max_value,violated_factor,critical_factor,equality_factor = 2.5*(10**4),2*10**4,10**6,10**3
    print(f"Data file path: {data_file}")
    data = pd.read_csv(data_file)
    fixture = csv_to_fixture(data)
    value = fixture_attractiveness(fixture, equality_factor, max_value)
    print(f"Value of AFL Fixture for {data_file}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the data file",
    )

    args = parser.parse_args()
    main(args.data_file)


