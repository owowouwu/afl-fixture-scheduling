import random
import numpy as np
from tqdm import tqdm
from collections import deque
from .local_search import iterated_local_search, random_neighbour

class GreedyScheduler:
    def __init__(self, tourn):
        self.tourn = tourn
        self.n_teams = len(tourn.teams)
        self.n_games_per_round = self.n_teams / 2
        self.n_rounds = tourn.rounds
        self.n_timeslots = len(tourn.timeslots)
        self.n_stadiums = len(tourn.stadiums)

    def find_rts(self, team1, team2, k, weight_matrix):
        # finds the top k round, timeslot, stadium configuration for a team based on weights
        x = weight_matrix[team1, team2, :, :, :]
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

    def adapt_greedy_soft(self, team1, team2, stadium, timeslot, round, weight_matrix,
                          timeslot_usage, stadium_usage, weight, update_usages = False
                          ):
        # adapt the weighting for greedy algorithm based on games played so far
        weight_matrix[team1, team2,:,:,:] = -np.inf
        weight_matrix[team2, team1, :, :, :] = -np.inf

        # can't play in same round again
        weight_matrix[team1, :, :, :, round] = -np.inf
        weight_matrix[team2, :, :, :, round] = -np.inf
        weight_matrix[:, team1, :, :, round] = -np.inf
        weight_matrix[:, team2, :, :, round] = -np.inf

        # restrictions on timeslots for each round
        if timeslot == 0 or timeslot == 1: # thurs or friday night can only have one game
            weight_matrix[:, :, :, timeslot, round] *= weight
            timeslot_usage[round][timeslot] += 1
        else:
            timeslot_usage[round][timeslot] += 1
            if timeslot_usage[round][timeslot] == 2: # all others can have two games
                weight_matrix[:, :, :, timeslot, round] *= weight

        # restrictions on the times a stadium can be used
        if timeslot in [2,3,4]:
            if update_usages: stadium_usage[round][0][stadium] += 1 # saturday usage
            if stadium_usage[round][0][stadium] == 1:
                weight_matrix[:,:,stadium,2, round] *= weight
                weight_matrix[:, :,stadium,3, round] *= weight
                weight_matrix[:, :,stadium,4, round] *= weight

        if timeslot in [5,6]:
            if update_usages: stadium_usage[round][1][stadium] += 1  # sunday usage
            if stadium_usage[round][1][stadium] == 1:
                weight_matrix[:, :, stadium, 5, round] *= weight
                weight_matrix[:, :, stadium, 6, round] *= weight

        return weight_matrix, timeslot_usage, stadium_usage


        pass
    def adapt_greedy(self, team1, team2, stadium, timeslot, round, weight_matrix,
                     timeslot_usage, stadium_usage):
        # adapt the weighting for greedy algorithm based on games played so far
        # can't play in exact same config
        weight_matrix[team1, team2,:,:,:] = -np.inf
        weight_matrix[team2, team1, :, :, :] = -np.inf

        # can't play in same round again
        weight_matrix[team1, :, :, :, round] = -np.inf
        weight_matrix[team2, :, :, :, round] = -np.inf
        weight_matrix[:, team1, :, :, round] = -np.inf
        weight_matrix[:, team2, :, :, round] = -np.inf

        # restrictions on timeslots for each round
        if timeslot == 0 or timeslot == 1: # thurs or friday night can only have one game
            weight_matrix[:, :, :, timeslot, round] = -np.inf
            timeslot_usage[round][timeslot] += 1
        else:
            timeslot_usage[round][timeslot] += 1
            if timeslot_usage[round][timeslot] == 2: # all others can have two games
                weight_matrix[:, :, :, timeslot, round] = -np.inf

        # restrictions on the times a stadium can be used
        if timeslot in [2,3,4]:
            stadium_usage[round][0][stadium] += 1 # saturday usage
            if stadium_usage[round][0][stadium] == 1:
                weight_matrix[:,:,stadium,2, round] = -np.inf
                weight_matrix[:, :,stadium,3, round] = -np.inf
                weight_matrix[:, :,stadium,4, round] = -np.inf

        if timeslot in [5,6]:
            stadium_usage[round][1][stadium] += 1  # sunday usage
            if stadium_usage[round][1][stadium] == 1:
                weight_matrix[:, :, stadium, 5, round] = -np.inf
                weight_matrix[:, :, stadium, 6, round] = -np.inf

        return weight_matrix, timeslot_usage, stadium_usage

    def generate_game_list(self, shuffle = False, include_extra = False):
        # generates a list of matches for each round
        # this is done by using a modified polygon method

        game_list = [
            [None for _ in range(self.n_teams // 2)] for _ in range(self.n_teams - 1)
        ]
        polygon = np.array(range(self.n_teams - 1))
        for i in range(self.n_teams - 1):
            game_list[i][0] = (polygon[0], 17)
            for j in range(1, self.n_teams // 2):
                game_list[i][j] = (polygon[j], polygon[self.n_teams - j - 1])

            if shuffle: random.shuffle(game_list[i])
            polygon = np.roll(polygon, 1)

        if shuffle: random.shuffle(game_list)

        if include_extra:
            n_extra_rounds = self.n_rounds - self.n_teams + 1
            extra_rounds = random.sample(game_list[0: (self.n_teams // 2)], n_extra_rounds)
            game_list += extra_rounds


        return game_list

    def generate_whole_fixture(self, rcl_length=10, seed=None, print_fixture=False, resolve_conflict=None):
        """
        generates a greedy fixture by allocating every match to a timeslot stadium and round, does not work well
        :param rcl_length:
        :param seed:
        :param print_fixture:
        :return:
        """
        if seed is not None:
            random.seed(seed)
        initial_rounds = self.n_teams - 1
        initial_fixture_matrix = np.zeros((self.n_teams, self.n_teams, self.n_stadiums, self.n_timeslots, self.n_teams))

        # shuffle games to select teams randomly
        game_list = self.generate_game_list(shuffle=True)
        print("Game list", game_list)
        games_left = [
            match for rounds in game_list for match in rounds
        ]
        games_to_play = len(games_left)
        games_left = deque(games_left)

        weight_matrix = self.tourn.weight_matrix.copy()
        weight_matrix_init = weight_matrix[:,:,:,:, 0:initial_rounds]
        soft_weight_matrix = weight_matrix_init.copy()
        # keep track of usage
        timeslot_usage = np.zeros((initial_rounds, self.n_timeslots))
        stadium_usage = np.zeros((initial_rounds, 2, self.n_stadiums))

        print("Developing initial rounds.\n")
        games_not_found = 0
        # construct schedule for first 18 rounds
        while games_to_play > 0:
            # choose team1, team2 (randomly since shuffled)
            match = games_left.pop()

            # shuffle home and away

            team1, team2 = match
            games_to_play -= 1

            # find stadium, round and timeslot by building a list of the best K

            candidates, scores = self.find_rts(team1, team2, k = rcl_length, weight_matrix=weight_matrix_init)
            # randomly sample among the best K
            try:
                assignment = random.choice(candidates)
                stadium, timeslot, r = assignment
            except:
                print(
                    f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, switching home and away")
                team1, team2 = team2, team1
                candidates, scores = self.find_rts(team1, team2, rcl_length, weight_matrix_init)
                try:
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment
                except:
                    print(f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, using soft weighting")
                    candidates, scores = self.find_rts(team1, team2, rcl_length, soft_weight_matrix)

            initial_fixture_matrix[team1, team2, stadium, timeslot, r] = 1
            weight_matrix_init,timeslot_usage, stadium_usage = self.adapt_greedy(team1, team2, stadium, timeslot, r,
                                                                            weight_matrix=weight_matrix_init,
                                                                            timeslot_usage=timeslot_usage,
                                                                            stadium_usage=stadium_usage
                                                                            )
            soft_weight_matrix, timeslot_usage, stadium_usage = self.adapt_greedy_soft(team1, team2, stadium, timeslot,r,
                                                                                       weight_matrix=soft_weight_matrix,
                                                                                       timeslot_usage=timeslot_usage,
                                                                                       stadium_usage=stadium_usage,
                                                                                       weight=0.01,
                                                                                       update_usages=False
                                                                                       )

            if print_fixture:
                self.tourn.print_round_game(team1, team2, stadium, timeslot, r)

        # next stage
        extra_rounds = self.n_rounds - initial_rounds
        game_list = self.generate_game_list(shuffle=True)
        games_left = [
            match for rounds in game_list for match in rounds
        ]
        games_to_play = extra_rounds * self.n_games_per_round
        games_left = deque(games_left)
        extra_fixture_matrix = np.zeros((self.n_teams, self.n_teams, self.n_stadiums, self.n_timeslots, extra_rounds))

        weight_matrix = self.tourn.weight_matrix.copy()
        weight_matrix_extra = weight_matrix[:, :, :, :, initial_rounds:]
        soft_weight_matrix = weight_matrix_extra.copy()
        # keep track of usage
        timeslot_usage_e = np.zeros((extra_rounds, self.n_timeslots))
        stadium_usage_e = np.zeros((extra_rounds, 2, self.n_stadiums))
        # construct schedule for the extra rounds

        print("\nDeveloping extra rounds\n")

        while games_to_play > 0:
            # choose team1, team2 (randomly since shuffled)
            match = games_left.pop()
            team1, team2 = match
            games_to_play -= 1
            candidates, scores = self.find_rts(team1, team2, k=rcl_length, weight_matrix=weight_matrix_extra)
            try:
                assignment = random.choice(candidates)
                stadium, timeslot, r = assignment
            except:
                print(
                    f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, switching home and away")
                team1, team2 = team2, team1
                candidates, scores = self.find_rts(team1, team2, rcl_length, weight_matrix_init)
                try:
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment
                except:
                    print(f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, using soft weighting")
                    candidates, scores = self.find_rts(team1, team2, rcl_length, soft_weight_matrix)

            extra_fixture_matrix[team1, team2, stadium, timeslot, r] = 1
            weight_matrix_extra,timeslot_usage_e, stadium_usage_e = self.adapt_greedy(team1, team2, stadium, timeslot, r,
                                                                            weight_matrix=weight_matrix_extra,
                                                                            timeslot_usage=timeslot_usage_e,
                                                                            stadium_usage=stadium_usage_e
                                                                            )
            soft_weight_matrix, timeslot_usage, stadium_usage = self.adapt_greedy_soft(team1, team2, stadium, timeslot,
                                                                                       r,
                                                                                       weight_matrix=soft_weight_matrix,
                                                                                       timeslot_usage=timeslot_usage,
                                                                                       stadium_usage=stadium_usage,
                                                                                       weight=0.01,
                                                                                       update_usages=False
                                                                                       )

            if print_fixture:
                self.tourn.print_round_game(team1, team2, stadium, timeslot, r)

        final_fixture = np.concatenate((initial_fixture_matrix, extra_fixture_matrix), axis = 4)
        stadium_usage = np.concatenate((stadium_usage, stadium_usage_e), axis = 0)
        timeslot_usage = np.concatenate((timeslot_usage, timeslot_usage_e), axis = 0)
        print("\nDone!\n")
        return final_fixture, timeslot_usage, stadium_usage


    def find_slot_for_team(self, team1, team2, round, weight_matrix, k):
        # finds the top k  timeslot, stadium configuration for a team based on weights
        x = weight_matrix[team1, team2, :, :, round]
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

    def generate_fixture_from_matchlist(self, rcl_length,seed = None, print_fixture = False, match_list = None):
        """
        generate a fixture given a list of matches by round which satisfies the DRR format
        :param match_list:
        :param seed:
        :return:
        """
        if seed is not None:
            random.seed(seed)
        if match_list is None:
            match_list = self.generate_game_list(shuffle=True, include_extra=True)
        initial_rounds = self.n_teams - 1
        initial_fixture_matrix = np.zeros((self.n_teams, self.n_teams, self.n_stadiums, self.n_timeslots, initial_rounds))
        # shuffle games to select teams randomly
        weight_matrix = self.tourn.weight_matrix.copy()
        weight_matrix_init = weight_matrix[:,:,:,:, 0:initial_rounds]
        soft_weight_matrix = weight_matrix_init.copy()
        # keep track of usage
        timeslot_usage = np.zeros((initial_rounds, self.n_timeslots))
        stadium_usage = np.zeros((initial_rounds, 2, self.n_stadiums))
        if print_fixture: print(f"Generating fixture for rounds {0}-{16}")
        for r in range(initial_rounds):
            matches_to_assign = match_list[r]
            for match in matches_to_assign:
                team1, team2 = match
                candidates, scores = self.find_slot_for_team(team1, team2, r, weight_matrix_init, rcl_length)
                try:
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment
                except:
                    print(f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, using soft weighting")
                    candidates, scores = self.find_slot_for_team(team1, team2, r, soft_weight_matrix, rcl_length)
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment

                initial_fixture_matrix[team1, team2, stadium, timeslot, r] = 1
                weight_matrix_init, timeslot_usage, stadium_usage = self.adapt_greedy(team1, team2, stadium, timeslot,r,
                                                                                      weight_matrix=weight_matrix_init,
                                                                                      timeslot_usage=timeslot_usage,
                                                                                      stadium_usage=stadium_usage
                                                                                      )
                soft_weight_matrix, timeslot_usage, stadium_usage = self.adapt_greedy_soft(team1, team2, stadium, timeslot,r,
                                                                                      weight_matrix=soft_weight_matrix,
                                                                                      timeslot_usage=timeslot_usage,
                                                                                      stadium_usage=stadium_usage,
                                                                                      weight = 0.01,
                                                                                      update_usages = False
                                                                                      )



                if print_fixture: self.tourn.print_round_game(team1, team2, stadium, timeslot, r)

        extra_rounds = self.n_rounds - initial_rounds
        extra_fixture_matrix = np.zeros((self.n_teams, self.n_teams, self.n_stadiums, self.n_timeslots, extra_rounds))

        weight_matrix = self.tourn.weight_matrix.copy()

        weight_matrix_extra = weight_matrix[:, :, :, :, initial_rounds:]
        soft_weight_matrix = weight_matrix_extra.copy()
        # keep track of usage
        timeslot_usage_e = np.zeros((extra_rounds, self.n_timeslots))
        stadium_usage_e = np.zeros((extra_rounds, 2, self.n_stadiums))
        if print_fixture: print(f"Generating fixture for extra rounds {17}-{21}")
        for r in range(extra_rounds):
            matches_to_assign = match_list[r + initial_rounds]
            for match in matches_to_assign:
                team1, team2 = match
                candidates, scores = self.find_slot_for_team(team1, team2, r, weight_matrix_extra, rcl_length)
                try:
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment
                except:
                    print(f"Did not find game - {self.tourn.cnames[team1]} vs. {self.tourn.cnames[team2]} being considered, using soft weighting")
                    candidates, scores = self.find_slot_for_team(team1, team2, r, soft_weight_matrix, rcl_length)
                    assignment = random.choice(candidates)
                    stadium, timeslot = assignment

                extra_fixture_matrix[team1, team2, stadium, timeslot, r] = 1
                weight_matrix_extra, timeslot_usage, stadium_usage = self.adapt_greedy(team1, team2, stadium, timeslot,r,
                                                                                      weight_matrix=weight_matrix_extra,
                                                                                      timeslot_usage=timeslot_usage_e,
                                                                                      stadium_usage=stadium_usage_e
                                                                                      )
                soft_weight_matrix, timeslot_usage, stadium_usage = self.adapt_greedy_soft(team1, team2, stadium, timeslot,r,
                                                                                      weight_matrix=soft_weight_matrix,
                                                                                      timeslot_usage=timeslot_usage,
                                                                                      stadium_usage=stadium_usage,
                                                                                      weight = 0.01,
                                                                                      update_usages = False
                                                                                      )

                if print_fixture: self.tourn.print_round_game(team1, team2, stadium, timeslot, r + initial_rounds)
        final_fixture = np.concatenate((initial_fixture_matrix, extra_fixture_matrix), axis = 4)
        stadium_usage = np.concatenate((stadium_usage, stadium_usage_e), axis = 0)
        timeslot_usage = np.concatenate((timeslot_usage, timeslot_usage_e), axis = 0)
        print("\nDone!\n")
        return final_fixture, timeslot_usage, stadium_usage

    def grasp_heuristic(self, seed, iterations, rcl_length = 10, greedy_constructor = 'by_round', do_ils = False, ils_iterations = 100,
                        progress_bar = False, max_value = 2*(10**4), violated_factor = 2*(10**4), critical_factor = 10**6, equality_factor = 10**3
                        ):
            random.seed(seed)
            best_obj = -np.inf
            best_schedule = None
            objective = lambda x: self.tourn.fixture_attractiveness(x, violated_factor=violated_factor,
                                                                    critical_factor=critical_factor,
                                                                    equality_factor = equality_factor,
                                                                    max_value = max_value
                                                                    )
            for i in tqdm(range(iterations), disable = (not progress_bar)):
                if greedy_constructor == 'by_round':
                    fixture, _, _ = self.generate_fixture_from_matchlist(rcl_length = rcl_length)
                elif greedy_constructor == 'whole_fixture':
                    fixture, _, _ = self.generate_whole_fixture(rcl_length=rcl_length)
                # fixture = self.fix_schedule(fixture) # fix schedule to maintainf easibility
                if do_ils:
                    fixture, new_obj = iterated_local_search(fixture, objective, neigh_fun=random_neighbour,
                                                                  max_it=ils_iterations)
                else:
                    new_obj = objective(fixture)
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_schedule = fixture

            return best_schedule, best_obj
