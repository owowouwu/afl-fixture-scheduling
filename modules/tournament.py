import numpy as np

class Tournament:

    def __init__(self, teams, locations, stadiums, timeslots, rounds):
        # initialise the data for teams and stadiums
         self.stadiums = list(stadiums.values())
         self.locations = locations
         self.teams = list(teams.values())
         self.timeslots = timeslots
         self.rounds = rounds

         self.cnames = list(teams.keys())
         self.snames = list(stadiums.keys())
         
         # rounds (R), timeslots (T), stadiums (S), competitors/teams (C), and rivals/enemies (E)
         self.R = range(rounds)
         self.T = range(len(timeslots))
         self.S = range(len(stadiums))
         self.C = range(len(teams))
         self.E = self._init_rivals_matrix()
         
         # convert home stadiums into integers

         for t in self.teams:
            t['home_stadiums'] = set([self.snames.index(s) for s in t['home_stadiums']])
            t['rivals'] = set([self.cnames.index(c) for c in t['rivals']])
        
         for t in self.teams:
            t['home_location_stadiums'] = []
        
         for i in self.C:
            for s in self.S:
                 if self.stadiums[s]['location'] == self.teams[i]['home_location']:
                    self.teams[i]['home_location_stadiums'].append(s)

         for l in self.locations:
            self.locations[l]['stadiums'] = set([self.snames.index(s) for s in self.locations[l]['stadiums']])

         self.home_location_stadiums = [t['home_location_stadiums'] for t in self.teams]
         self.home_stadiums = [t['home_stadiums'] for t in self.teams]
         self.fixture_matrix = np.array([[[[[0 for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C], dtype='bool')
         self.attractiveness_matrix = np.array([[[[[self.attractiveness(i,j,s,t,r) for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C])
         self.weight_matrix = np.array([[[[[self.prelim_attractiveness(i,j,s,t,r) for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C])

    def _init_rivals_matrix(self):
        return [
            [1 if team2 in self.teams[team1]['rivals'] else 0 for team2 in self.C] for team1 in self.C
        ]

    # utilities

    def print_fixture(self):
        for r in self.R:
            print(f"Round {r}:")
            self.print_round(r)
        

    def print_round(self, r):
        for t in self.T:
            self.print_timeslot(r,t)
        

    def print_timeslot(self, r, t):
        game_played = np.array(np.where(self.fixture_matrix[:,:,:,t,r] > 0))
        if game_played.shape[1] == 0:
            print(f"{self.timeslots[t]['name']}: No Game")
            return
        for j in range(game_played.shape[1]):
            home = game_played[0][j]
            away = game_played[1][j]
            stad = game_played[2][j]
            self.print_game(home, away, stad, t)
        
    def print_round_game(self, i, j,s, t,r):
        print(f"Round {r}, {self.timeslots[t]['name']}: {self.cnames[i]} vs. {self.cnames[j]} at {self.snames[s]}")

    def print_game(self, i, j, s, t):
        print(f"{self.timeslots[t]['name']}: {self.cnames[i]} vs. {self.cnames[j]} at {self.snames[s]}")
    
    def prelim_attractiveness(self, i, j, s, t, r):
        # attractiveness but also considers some simple violated constraints
        # adjust as needed
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        sigma = 1.0
        xi = 1.0
        score = 1
        if not s in self.teams[i]['home_stadiums']:
            score = -np.inf

        if i == j:
            score = -np.inf
        
    
        if score < 0: return score
        
        if self.E[i][j]:
            score *= 1+alpha
        
        score /= max(abs(self.teams[i]['ranking'] - self.teams[j]['ranking']),1)
        score /= (self.teams[i]['ranking'] + self.teams[j]['ranking']) / 2
        
        if s in self.teams[j]['home_stadiums']:
            score *= (1+beta)


        score *=  self.stadiums[s]['size']
        score *= (self.teams[i]['fans'] + self.teams[j]['fans'])
        
        score *= self.timeslots[t]['value']
        
        return score


    def probability_win(self, i, j, s):
        probability = self.teams[i]['wins']/(self.teams[i]['wins']+self.teams[j]['wins'])
        if s not in self.teams[j]['home_stadiums']:
            probability += (1-probability)/2.5
        elif s not in self.teams[j]['home_location_stadiums']:
            probability += (1-probability)/4
        else:
            probability += (1-probability)/10
            
        return probability

    def expected_win_variance(self, fixture):
        results = []
        expected_wins = [0]*18
        for r in self.R:
            for i in self.C:
                for j in self.C:
                    for s in self.S:
                        for t in self.T:
                            expected_wins[i] += self.probability_win(i, j, s)*fixture[i][j][s][t][r]
                            expected_wins[i] += (1-self.probability_win(j, i, s))*fixture[j][i][s][t][r]
            
            results.append(np.var(expected_wins))
        
       
        return sum((i+1)*results[i] for i in range(len(results)))

    def attractiveness(self, i, j, s, t, r):
        # adjust as needed
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        sigma = 1.0
        xi = 1.0
        score = 1
        
        if self.E[i][j]:
            score *= 1+alpha
        
        score /= max(abs(self.teams[i]['ranking'] - self.teams[j]['ranking']),1)
        score  /= (self.teams[i]['ranking'] + self.teams[j]['ranking'])/2
        
        if s in self.teams[j]['home_stadiums']:
            score *= (1+beta)
            
        score *=  self.stadiums[s]['size']
        score *= (self.teams[i]['fans'] + self.teams[j]['fans'])
        
        score *= self.timeslots[t]['value']
        
        return score

    def fixture_attractiveness(self, fixture,max_value = 10000,violated_factor = 0,critical_factor = 0, equality_factor = 0):
        total_score = 0
        
        for r in self.R:
            for t in self.T:
                value = 0
                for i in self.C:
                    for j in self.C:
                        for s in self.S:
                            if fixture[i,j,s,t,r] == 1:
                                value += self.attractiveness(i, j, s, t, r)
                                
                total_score += min(max_value,value)
                
        violated, critical = self.feasibility(fixture)
        equality = equality_factor*self.expected_win_variance(fixture)
        return total_score - violated_factor*violated - critical_factor*critical - equality

    def feasibility(self, fixture, debug=False):
        violated = 0
        critical = 0
        

        for i in self.C:
            n_home_games = sum(fixture[i, j, s, t, r] for j in self.C for s in self.S for t in self.T for r in self.R)
            if n_home_games != 11: # 11 home games
                critical += abs(n_home_games - 11) # Number of home games
                if debug:
                    print(f"Violated critical constraint for total number of home games for team {self.cnames[i]}. Played {n_home_games} home games.")

            for r in self.R:
                if sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for j in self.C for s in self.S for t in self.T) == 0:
                    critical += 1
                    if debug: print(f"Violated critical constraint - {self.cnames[i]} did not play during round {r}")

            home_stadium_diff = list(set(self.S).difference(self.teams[i]['home_stadiums']))
            n_games_outside_home = sum(fixture[i,j,s,t,r] for j in self.C for s in home_stadium_diff for t in self.T for r in self.R)
            if n_games_outside_home > 0:
                critical += n_games_outside_home
                if debug: print(f"Violated critical constraint - team {self.cnames[i]} played {n_games_outside_home} games outside home.")
                                                            

            # violated += sum(fixture[i, j, s, t, r] 
            #                         for j in self.C 
            #                         for s in list(set(self.S).difference(self.teams[i]['home_stadiums'])) 
            #                         for t in self.T for r in self.R) # Home games outside home ground

            if sum(fixture[i, i, s, t, r] for s in self.S for t in self.T for r in self.R) > 0:
                critical += vs
                if debug: print(f"Violated critical constraint - {self.cnames[i]} is listed as playing itself.")

            for j in self.C:
                if j == i: continue
                vs = sum(fixture[i, j, s, t, r] + fixture[j,i,s,t,r] for s in self.S for t in self.T for r in self.R)
                if vs == 0:
                    violated += 1 # Don't play the other team
                    if debug:
                        print(f"Violated constraint - {self.cnames[i]} did not play {self.cnames[j]}")
                elif vs > 2:
                    violated+= vs-1 # Play the other team too much
                    if debug:
                        print(f"Violated constraint - {self.cnames[i]} played {self.cnames[j]} {vs} times")
            
                for r in self.R[:-1]:
                    played_early = max(0,sum(fixture[i][j][s][t][r] + fixture[j][i][s][t][r] for s in self.S for t in [5,6])+ 
                                sum(fixture[i, j, s, t, r+1]+fixture[j, i, s, t, r+1] for s in self.S for t in [0]) - 1)
                    if played_early:
                        print(f"Violated constraint - {self.cnames[i]} played {self.cnames[j]} again too soon.")

            for r in self.R[:-2]:
                if sum(fixture[j][i][s][t][r_]+fixture[i][j][s][t][r_] for j in self.C for s in self.teams[i]['home_location_stadiums']
                                 for t in self.T for r_ in range(r,r+3)) == 0:
                    violated += 1
                    if debug: print(f"Violated constraint - {self.cnames[i]} played consecutive rounds rounds {r},{r+1},{r+2} outside of home location")

            for r in self.R[:-3]:
                if sum(fixture[i, j, s, t, r_] for j in self.C for s in self.T for t in self.T for r_ in range(r,r+4)) == 0:
                    violated += 1
                    if debug: print(f"Violated constraint - {self.cnames[i]} played consecutive rounds {r},{r+1},{r+2},{r+3} as away games")


                
        for r in self.R:
            for s in self.S:
                if sum(fixture[i, j, s, t, r] for i in self.C for j in self.C for t in [2,3,4]) > 1: # 2+ Saturday games in stadium
                    violated += 1
                    if debug:
                        print(f"Violated critical constraint - 2+ Saturday games at stadium {self.snames[s]} in round {r}")
                    
                if sum(fixture[i, j, s, t, r] for i in self.C for j in self.C for t in [5,6]) > 1: # 2+ Sunday games in stadium
                    violated += 1
                    if debug:
                        print(f"Violated critical constraint - 2+ Sunday games at stadium {self.snames[s]} in round {r}")

        # Constraint: No more than two games in any timeslot, and only one on Thursday and Friday night, at least one in each
        for r in self.R:
            
            for t in [2,3,4,5,6]:
                
                if sum(fixture[i][j][s][t][r] for i in self.C for j in self.C for s in self.S) > 2:
                    violated += 1
                    if debug: print(f"Violated constraint - more than 2 games played on {self.timeslots[t]['name']} in round {r}")
    

        return violated, critical

    def find_stadium(self,fixture, team, t, r):
        # finds a stadium for a home team in round r timeslot t
        for stad in self.teams[team]['home_stadiums']:
            if np.sum(fixture[:,:,stad,t,r]) == 0:
                return stad

        # can't find a stadium then find a stadium that is at the very least in the same state
        for stad in self.locations[self.teams[team]['home_location']]['stadiums']:
            if np.sum(fixture[:,:,stad,t,r]) == 0:
                return stad
        
        # no stadium can be found
        return -1

    def find_game_to_swap(self,fixture,i,j):
        games = np.array(np.where(fixture[i,j,:,:,:] == 1))
        for i in range(games.shape[1]):
            s = games[0][i]
            t = games[1][i]
            r = games[2][i]
            s_new = self.find_stadium(fixture, j, t, r)
            if s != -1: return True, (s,t,r), s_new
        
        return False, False, False
        

    def home_away_swap(self, t, r):
        # set the current game being played to 0
        new_fixture = self.fixture_matrix.copy()
        game_played = np.array(np.where(new_fixture[:,:,:,t,r] == 1))
        h = game_played[0][0]
        a = game_played[1][0]
        s = game_played[2][0]
        new_fixture[h,a,s,t,r] = 0
        # now we need to find the stadium for the new home team
        s = self.find_stadium(new_fixture, a, t, r)
        # no swap can be found
        if s == -1:
            new_fixture[h,a,s,t,r] = 1
            return new_fixture
        new_fixture[a,h,s,t,r] = 1

        return new_fixture

    def team_swap(self, c1, c2):
        new_fixture = self.fixture_matrix.copy()
        team1_home = np.array(np.where(new_fixture[c1,:,:,:,:] == 1)).T
        team1_away = np.array(np.where(new_fixture[:,c1,:,:,:] == 1)).T
        team2_home = np.array(np.where(new_fixture[c2,:,:,:,:] == 1)).T
        team2_away = np.array(np.where(new_fixture[:,c2,:,:,:] == 1)).T

        
        # away games are easy
        if team1_away.shape[0] > 0:
            for game in team1_away:
                h = game[0]
                s = game[1]
                t = game[2]
                r = game[3]
                new_fixture[h,c1,s,t,r] = 0
                new_fixture[h,c2,s,t,r] = 1
        
        if team2_away.shape[0] > 0:
            for game in team2_away:
                h = game[0]
                s = game[1]
                t = game[2]
                r = game[3]
                new_fixture[h,c2,s,t,r] = 0
                new_fixture[h,c1,s,t,r] = 1

        if team1_home.shape[0] > 0:
            for game in team1_home:
                a = game[0]
                s = game[1]
                t = game[2]
                r = game[3]
                new_fixture[c1,a,s,t,r] = 0
                s = self.find_stadium(new_fixture, c2, t, r)
                # can't swap
                if s == -1:
                    new_fixture[c1,a,s,t,r] = 1
                    continue
                new_fixture[c2,a,s,t,r] = 1
        
        if team2_home.shape[0] > 0:
            for game in team2_home:
                a = game[0]
                s = game[1]
                t = game[2]
                r = game[3]
                new_fixture[c2,a,s,t,r] = 0
                s = self.find_stadium(new_fixture, c1, t, r)
                if s == -1:
                    new_fixture[c2,a,s,t,r] = 1
                    continue
                new_fixture[c1,a,s,t,r] = 1

        return new_fixture
    
    def fix_fixture(self, fixture):
        # applies a couple quick fixes to a poor fixture to increase its feasibility
        self.balance_home_games(fixture)

        return fixture
        

    def balance_home_games(self, fixture):
        
        home_games_per_team = np.array([
            np.sum(fixture[i,:,:,:,:]) for i in self.C
        ])
        # sort home games
        teams = np.argsort(home_games_per_team)
        home_games_per_team = home_games_per_team[teams]
        for i in self.C:
            if home_games_per_team[i] == 11: break

            for j in range(len(teams) - 1, -1, - 1):
                if i == j: break
                if home_games_per_team[j] == 11: continue
                while home_games_per_team[i] < 11:
                    found_game, game, s_new = self.find_game_to_swap(fixture, teams[j],teams[i])
                    if not found_game: break
                    s,t,r = game
                    fixture[teams[j], teams[i], s, t, r] = 0
                    fixture[teams[i], teams[j], s_new, t, r] = 1
                    home_games_per_team[i] += 1
                    home_games_per_team[j] -= 1
                
        return fixture
    