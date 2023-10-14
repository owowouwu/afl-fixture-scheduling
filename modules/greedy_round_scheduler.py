import numpy as np
from tqdm import tqdm
from .local_search import IteratedLocalSearch

class GreedyByRoundScheduler:

    def __init__(self, tourn):
        self.tourn = tourn

    def find_best_inround(self, attractiveness_matrix, round = None):
        if round is not None:
            x = attractiveness_matrix[:,:,:,:,round]
        else:
            x = attractiveness_matrix
        
        best = tuple(np.unravel_index(np.argmax(x), x.shape))
        val = x[best]

        return best,val

    def find_k_best_inround(self, attractiveness_matrix, k, round = None, ):
        if round is not None:
            x = attractiveness_matrix[:,:,:,:,round]
        else:
            x = attractiveness_matrix
        flat = x.flatten()

        # Find the indices of the N largest values in the flattened array
        indices = np.argpartition(flat, -k)[-k:]
        # Convert the flattened indices back to the indices in the original matrix
        candidates = [tuple(np.unravel_index(idx, x.shape)) for idx in indices]
        scores = np.array([x[idx] for idx in candidates])
        to_discard = scores < 0
        scores = [scores[i] for i in range(k) if not to_discard[i]]
        candidates = [candidates[i] for i in range(k) if not to_discard[i]]
        
        return candidates, scores

    
    def adapt_greedy(self,attractiveness_matrix, tmp_matrix, round, game_chosen):
        
        pass

    def maintain_feasibility(self,tmp_matrix, game_chosen,
                            fixture_matrix = None, 
                            attractiveness_matrix = None):
        team1, team2, stadium, timeslot = game_chosen

        # teams cannot play games in the same round
        tmp_matrix[team1, :, :, :] = -np.inf
        tmp_matrix[:,team1,:,:] = -np.inf
        tmp_matrix[:, team2, :, :] = -np.inf
        tmp_matrix[team2,:,:,:] = -np.inf

        # check if a stadium has already been used two times in a day
        if timeslot in [2,3,4]:
            tmp_matrix[:,:,stadium,2] = -np.inf
            tmp_matrix[:,:,stadium,3] = -np.inf
            tmp_matrix[:,:,stadium,4] = -np.inf
        if timeslot in [5,6]:
            tmp_matrix[:,:,stadium,5] = -np.inf
            tmp_matrix[:,:,stadium,6] = -np.inf

        if fixture_matrix is not None:
            # max home games reached for team 1
            if np.sum(fixture_matrix[team1, : :, :, :]) >= 11:
                #print(f"limit of home games reached for team {self.tourn.cnames[team1]}")
                attractiveness_matrix[team1, :, : ,: ,:] *= 0.01 # need  to tune this
                # tmp_matrix[team1, :, :, :] *= 0.4
            # max away games reached for team 2
            if np.sum(fixture_matrix[:, team2, : , :, :]) >= 11:
                #print(f"limit of away games reached for team {self.tourn.cnames[team2]}")
                attractiveness_matrix[:, team2, :, : , :] *= 0.01
                # tmp_matrix[:, team2, :, :] *= 0.4
        
            attractiveness_matrix[team1,team2,:,:,:] = -np.inf
            attractiveness_matrix[team2,team1,:,:,:] = -np.inf
        
        if fixture_matrix is not None:
            return tmp_matrix, attractiveness_matrix
        return tmp_matrix

    def construct_greedy_round(self,attractiveness_matrix, round, timeslots, track_games = False,
                                fixture_matrix = None, print_games = False):
        games = [None for _ in range(timeslots)]
        games_fulfilled = 0
        tmp_matrix = attractiveness_matrix[:,:,:,:,round]

        # matrix to return to keep track of played games between two teams
        if track_games: return_matrix = attractiveness_matrix


        while games_fulfilled < timeslots:
            new_game, score = self.find_best_inround(tmp_matrix)
            # invalid game, consider it as a bye
            games_fulfilled += 1
            if score == -np.inf:
                games[timeslot] = [-1,-1,-1]
                if print_games: self.tourn.print_timeslot(round, timeslot)
                continue
            
            team1, team2, stadium, timeslot = new_game
            if print_games: self.tourn.print_game(team1, team2, stadium, timeslot)
            games[timeslot] = [team1,team2,stadium]
            # if we are running this in a loop to construct the whole schedule we want to keep track of the games played
            # between teams across all rounds
            
            if fixture_matrix is not None: 
                fixture_matrix[team1, team2, stadium, timeslot, round] = 1
                if track_games: 
                    tmp_matrix, return_matrix = self.maintain_feasibility(tmp_matrix, new_game,
                                                                    fixture_matrix, attractiveness_matrix)
                else:
                    tmp_matrix = self.maintain_feasibility(tmp_matrix=tmp_matrix, game_chosen=new_game)           
            else:
                tmp_matrix = self.maintain_feasibility(tmp_matrix=tmp_matrix, game_chosen=new_game)
        
        if track_games:
            if fixture_matrix is not None:
                return games, return_matrix, fixture_matrix
            return games, return_matrix
        if fixture_matrix is not None:
            return games, fixture_matrix
        return games

    def construct_greedy_schedule(self,attractiveness_matrix, rounds, timeslots):
        tmp_matrix = attractiveness_matrix.copy()
        fixture = np.zeros(shape = attractiveness_matrix.shape)
        schedule = [None for _ in range(rounds)]
        for round in range(rounds):
            print(f"Round {round}: ")
            games, tmp_matrix, fixture = self.construct_greedy_round(tmp_matrix, round, timeslots, 
                                                                track_games=True,
                                                                fixture_matrix=fixture)
            schedule[round] = games
        
        return schedule, fixture


    def construct_greedy_round_random(self, attractiveness_matrix, round, n_games, timeslots,
                                      rcl_length = 10,
                                      track_games = False,
                                      fixture_matrix = None,
                                      print_games = False):
        games_fulfilled = 0
        tmp_matrix = attractiveness_matrix[:,:,:,:,round]
        timeslots_left = set(range(n_games))
        games_per_timeslot = np.zeros(timeslots)
        # matrix to return to keep track of played games between two teams
        if track_games: return_matrix = attractiveness_matrix
        while games_fulfilled < n_games:
            candidate_list, scores = self.find_k_best_inround(tmp_matrix, k = rcl_length)

            games_fulfilled += 1
            n_candidates = len(candidate_list)
            # if max games for timeslots reached delete for consideration
            # this is done by setting the attractiveness to -1


            # select at random
            to_select = np.random.choice(n_candidates)
            new_game = candidate_list[to_select]
            
            team1, team2, stadium, timeslot = new_game
            # check max games per timeslot
            games_per_timeslot[timeslot] += 1
            if timeslot == 0 or timeslot == 1: # thrus fri night only one game
                tmp_matrix[:,:,:,timeslot] = -np.inf
            else:
                if games_per_timeslot[timeslot] == 2:
                    tmp_matrix[:, :, :, timeslot] = -np.inf

            if print_games: self.tourn.print_game(team1, team2, stadium, timeslot)
            if fixture_matrix is not None: 
                fixture_matrix[team1, team2, stadium, timeslot, round] = 1
                if track_games: 
                    tmp_matrix, return_matrix = self.maintain_feasibility(tmp_matrix, new_game,
                                                                    fixture_matrix, attractiveness_matrix)
                else:
                    tmp_matrix = self.maintain_feasibility(tmp_matrix=tmp_matrix, game_chosen=new_game)           
            else:
                tmp_matrix = self.maintain_feasibility(tmp_matrix=tmp_matrix, game_chosen=new_game)
        
        if track_games:
            if fixture_matrix is not None:
                return  return_matrix, fixture_matrix
            return  return_matrix
        if fixture_matrix is not None:
            return  fixture_matrix

    

    def construct_greedy_schedule_random(self,attractiveness_matrix,
                                         rounds, n_games, timeslots, rcl_length = 10, print_games = False):
        tmp_matrix = attractiveness_matrix.copy()
        fixture = np.zeros(shape = attractiveness_matrix.shape)

        for round in range(rounds):
            if print_games: print(f"Round {round}: ")
            tmp_matrix, fixture = self.construct_greedy_round_random(tmp_matrix, round,
                                                                            n_games=n_games,
                                                                            timeslots=timeslots,
                                                                            rcl_length=rcl_length,
                                                                            track_games=True,
                                                                            fixture_matrix=fixture,
                                                                            print_games = print_games)

            if round == 17:
                tmp_matrix = attractiveness_matrix.copy()
        
        return fixture

    def grasp_heuristic(self, iterations, local_it, rcl_length, 
                        attractiveness_matrix, rounds, timeslots):
        
        best_obj = -np.inf
        best_schedule = None
        violated_penalty = 1000
        critical_penalty = 1000000
        objective = lambda x: self.tourn.fixture_attractiveness(x, violated_factor = violated_penalty, critical_factor = critical_penalty)
        for i in tqdm(range(iterations)):
            arr = attractiveness_matrix.copy()
            schedule, fixture = self.construct_greedy_schedule_random(
                arr, rounds=rounds, timeslots = timeslots, rcl_length = rcl_length
            )
            #fixture = self.fix_schedule(fixture) # maintain feasibility
            new_schedule, new_obj = iterated_local_search(fixture, objective, neigh_fun = random_neighbour, max_it = local_it)
            if new_obj > best_obj:
                best_obj = new_obj
                best_schedule = new_schedule

        return best_schedule, best_obj
