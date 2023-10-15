import numpy as np
from numpy import random as r
from itertools import combinations
from .tournament import Tournament

class IteratedLocalSearch:

    def __init__(self, tourn):
        self.tourn = tourn

    def search(self, initial_schedule, obj_fun, max_it = 10000, stop_criterion = 0.005):
        best_obj = obj_fun(initial_schedule)
        best_schedule = initial_schedule
        curr_schedule = initial_schedule
        curr_obj = best_obj
        for t in range(max_it):
            new_schedule = self.random_neighbour(curr_schedule)
            new_obj = obj_fun(new_schedule)
            if new_obj > curr_obj:
                curr_obj = new_obj
                curr_schedule = new_schedule
            
            if curr_obj > best_obj:
                stop = (curr_obj - best_obj) / best_obj < stop_criterion
                best_obj = curr_obj
                best_schedule = curr_schedule
                if stop: break

        return best_schedule,best_obj


    #A schedule is  a 5d array, where S[i,j,s,t,r] is a boolean that indicates whether team i plays team j in stadium k at timeslot l in round m

    def find_stadium(self,fixture, i, t, r):
        # finds a stadium for a home team in round r timeslot t
        for stad in self.tourn.home_stadiums[i]:
            if np.sum(fixture[:,:,stad,t,r]) == 0:
                return stad

        # can't find a stadium then find a stadium that is at the very least in the same state
        for stad in self.tourn.home_location_stadiums:
            if np.sum(fixture[:,:,stad,t,r]) == 0:
                return stad
            
        # no stadium can be found
        return -1


    def random_neighbour(self,schedule):
        #Function that gets a random neighbour, choosing which neighbourhood to explore with a particular probability
        #Parameters for tuning likeliness of using a particular neighbourhood function
        a = 0.33
        b = 0.66
        p = r.rand()
        if p<a:
            print('Swapping a home and away')
            new_schedule = self.random_neighbour_home_swap(schedule)
        elif p>=a and p<b:
            print('Moving a match')
            new_schedule = self.random_neighbour_match_move(schedule)
        else:
            print('Swapping a double-play')
            new_schedule = self.random_neighbour_double_swap(schedule)
        
        return new_schedule
        
    def random_neighbour_home_swap(self,schedule):
        #Function that swaps a random match from home to away
        new_schedule = schedule.copy()
        
        #Pick two teams
        i, j = r.choice(range(0,18), 2, replace=False)

        #Choose one of the one plus games they play
        #Aight, so this returns a list of indices, each should be (i,j, non ij index of any actual ij matches)
        i_home = [(i,j) + tuple(index) for index in zip(*np.nonzero(schedule[i,j,:,:,:]))]
        j_home = [(j,i) + tuple(index) for index in zip(*np.nonzero(schedule[j,i,:,:,:]))]
        
        #Concatenate the two lists of indices for i vs j matches
        all_matches = i_home+j_home
        #print(all_matches)
        
        #Pick one
        if len(all_matches) <= 0:
            #Teams don't play - a constraint violation, but can occur. In this case, return the schedule unchanged
            return new_schedule
        old_match_index = r.randint(len(all_matches))
        old_match = all_matches[old_match_index]
        
        #Take the old match off the schedule
        new_schedule[old_match] = 0
        
        #Flip who plays at home
        new_match = list(old_match)
        new_match[0], new_match[1] = new_match[1], new_match[0]
        
        #Now find home stadium
        s = self.find_stadium(new_schedule, new_match[0], new_match[3], new_match[4])
        if s == -1:
            #Couldn't find home stadium, make no change
            new_schedule[old_match] = 1
            return new_schedule
        else:
            new_match[2] = s
            new_match = tuple(new_match)
            new_schedule[new_match] = 1

            #Return our new schedule
            return new_schedule
        
    def random_neighbour_match_move(self,schedule):
        #Function that moves a random match to a different time
        new_schedule = schedule.copy()
        
        #Pick hometeam
        i = r.randint(18)
        #Gets all the homegames the team plays
        homegames = [[i] + list(index) for index in zip(*np.nonzero(schedule[i,:,:,:,:]))]
        #Pick one
        old_match = homegames[r.randint(len(homegames))]
        
        
        done = False
        while done == False:
            #Pick a new timeslot and round
            t = r.randint(7)
            round = r.randint(22)
            
            #If noone else is playing in this stadium at this time, we move the match here
            if np.sum(schedule[:,:,old_match[2],t,round]) == 0:
                done = True
                new_schedule[old_match[0], old_match[1], old_match[2], t, round] = 1
                new_schedule[tuple(old_match)] = 0
        #Return the new schedule
        return new_schedule
        
        
        
    def random_neighbour_double_swap(self,schedule):
        #Function that takes two random double matches and swaps two of the teams between matches
        new_schedule = schedule.copy()
        #Find two pairs of pairs of teams that play twice
        done = False
        while done == False:
            #pick two teams
            i, j, k, l = r.choice(range(0,18), 4, replace=False)
            if np.sum(schedule[i,j,:,:,:])+np.sum(schedule[j,i,:,:,:]) == 2 and np.sum(schedule[k,l,:,:,:]) + np.sum(schedule[l,k,:,:,:])  == 2:
                done = True
        
        #Find all the matches i  and j play together
        i_home = [(i,j) + tuple(index) for index in zip(*np.nonzero(schedule[i,j,:,:,:]))]
        j_home = [(j,i) + tuple(index) for index in zip(*np.nonzero(schedule[j,i,:,:,:]))]
        i_j_matches = i_home + j_home
        
        #Find all the matches k and l play together
        k_home = [(k,l) + tuple(index) for index in zip(*np.nonzero(schedule[k,l,:,:,:]))]
        l_home = [(l,k) + tuple(index) for index in zip(*np.nonzero(schedule[l,k,:,:,:]))]
        k_l_matches = k_home + l_home
        
        #Picks one of the i vs j matches, and one of the k vs l matches
        i_j_to_swap = i_j_matches[r.randint(len(i_j_matches))]
        k_l_to_swap = k_l_matches[r.randint(len(k_l_matches))]
        
        #Gets indices of swapped matches - We're swapping the away teams of the two matches
        i_l_match = (i_j_to_swap[0], l) + i_j_to_swap[2:]
        k_j_match = (k_l_to_swap[0], j) + k_l_to_swap[2:]

        
        #Performs the swap
        new_schedule[i_j_to_swap] = 0
        new_schedule[k_l_to_swap] = 0
        new_schedule[i_l_match] = 1
        new_schedule[k_j_match] = 1
        
        #Return the new schedule
        return new_schedule