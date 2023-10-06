import numpy as np
from numpy import random as r
from itertools import combinations

def iterated_local_search(initial_schedule, obj_fun, neigh_fun, max_it = 10000, stop_criterion = 0.005):
    best_obj = obj_fun(initial_schedule)
    best_schedule = initial_schedule
    curr_schedule = initial_schedule
    curr_obj = best_obj
    for t in range(max_it):
        new_schedule = neigh_fun(curr_schedule)
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



def simulated_annealing_solve(initial_schedule, cooling_type, starting_temp, final_temp, cooling_size):
    #Establish cooling schedule
    if cooling_type == 'geometric':
        #Geometric cooling for lazy - Set an initial T, a geometric factor, and a size
        initial_T = starting_temp
        decay = (final_temp/starting_temp)**(1/cooling_size)
        cooling_schedule = [initial_T*decay**i for i in range(0,cooling_size)]
        #print(cooling_schedule)

    current_schedule = initial_schedule
    current_objective = objective(current_schedule)
    best_schedule = current_schedule
    best_objective = objective(current_schedule)

    #NB the way of doing simulated annealing in the lecture notes is a bit off, we just loop on T, rather than looping on T and nested loop of certain number iterations at each T
    for T in cooling_schedule:
        #Currently very inefficient - calling the whole objective function each time, rather than having random_neighbourhood return a delta objective value
        new_schedule = random_neighbourhood(current_schedule)
        new_objective = objective(new_schedule)
        
        if new_objective < current_objective or r.random() <= np.exp(-(new_objective-current_objective)/T):
            current_schedule = new_schedule
            current_objective = new_objective
            
            if current_objective < best_objective:
                best_objective = current_objective
                best_schedule = current_schedule

    return best_schedule, best_objective

#A schedule is  a 5d array, where S[i,j,s,t,r] is a boolean that indicates whether team i plays team j in stadium k at timeslot l in round m

def random_neighbour(schedule):
    #Function that gets a random neighbour, choosing which neighbourhood to explore with a particular probability
    #Parameters for tuning likeliness of using a particular neighbourhood function
    a = 0.33
    b = 0.66
    p = r.rand()
    if p<a:
        return schedule
        #new_schedule = random_neighbour_home_swap(schedule)
    elif p>=a and p<b:
        new_schedule = random_neighbour_match_move(schedule)
    else:
        new_schedule = random_neighbour_double_swap(schedule)
    
    return new_schedule
    
def random_neighbour_home_swap(schedule):
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
    print(all_matches)
    
    #Pick one
    old_match_index = r.randint(len(all_matches))
    old_match = all_matches[old_match_index]
    
    #Flip who plays at home
    new_match = list(old_match)
    new_match[0], new_match[1] = new_match[1], new_match[0]
    new_match = tuple(new_match)
    new_schedule[new_match] = 1
    new_schedule[old_match] = 0
    
    #Return our new schedule
    return new_schedule
    
def random_neighbour_match_move(schedule):
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
    
    
    
def random_neighbour_double_swap(schedule):
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
    