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
         # convert home stadiums into integers
         for t in self.teams:
            t['home_stadiums'] = set([self.snames.index(s) for s in t['home_stadiums']])
            t['rivals'] = set([self.cnames.index(c) for c in t['rivals']])
        
         

         for l in self.locations:
            self.locations[l]['stadiums'] = set([self.snames.index(s) for s in self.locations[l]['stadiums']])

        # rounds (R), timeslots (T), stadiums (S), competitors/teams (C), and rivals/enemies (E)
         self.R = range(rounds)
         self.T = range(len(timeslots))
         self.S = range(len(stadiums))
         self.C = range(len(teams))
         self.E = self._init_rivals_matrix()
         self.fixture_matrix = np.array([[[[[0 for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C], dtype='bool')
         self.attractiveness_matrix = np.array([[[[[self.attractiveness(i,j,s,t,r) for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C])
         self.prelim_attractiveness_matrix = np.array([[[[[self.prelim_attractiveness(i,j,s,t,r) for r in self.R] for t in self.T] for s in self.S] for j in self.C] for i in self.C])

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
        home = game_played[0][0]
        away = game_played[1][0]
        stad = game_played[2][0]
        self.print_game(home, away, stad, r, t)
        

    def print_game(self, i, j, s, r, t):
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
            score *= 0

        if i == j:
            score *= 0
        
    
        if score == 0: return score
        
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

    def fixture_attractiveness(self, fixture,max_value = 10000,violated_factor = 0,critical_factor = 0):
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
        return total_score - violated_factor*violated - critical_factor*critical

    def feasibility(self, fixture):
        violated = 0
        critical = 0
        
        if sum(fixture[i, j, s, t, r] for i in self.C for j in self.C for s in self.S for t in self.T for r in self.R) != 9*22:
            critical += 1
            violated += 1 # Number of total matches
            
        for i in self.C:
            if sum(fixture[i, j, s, t, r] for j in self.C for s in self.S for t in self.T for r in self.R) != 11: # 11 home games
                violated += 1 # Number of home games
            
            violated += sum(fixture[i, j, s, t, r] 
                                    for j in self.C 
                                    for s in list(set(self.S).difference(self.teams[i]['home_stadiums'])) 
                                    for t in self.T for r in self.R) # Home games outside home ground
            
            for j in self.C:
                vs = sum(fixture[i, j, s, t, r] for s in self.S for t in self.T for r in self.R)
                if i == j:
                    violated += vs # Can't play yourself
                    critical += vs
                
                else:
                    if vs == 0:
                        violated += 1 # Don't play the other team
                    elif vs > 2:
                        violated+= vs-1 # Play the other team too much
            
            last = 0
            for r in self.R:
                current = sum(fixture[i, j, s, t, r] 
                            for j in self.C 
                            for s in list(set(self.S).difference(self.teams[i]['home_stadiums']))
                            for t in self.T)
                violated += last*current # Two games in a row outside home location
                last = current

                
        for r in self.R:
            for s in self.S:
                if sum(fixture[i, j, s, t, r] for i in self.C for j in self.C for t in [2,3,4]) > 1: # 2+ Saturday games in stadium
                    violated += 1
                    critical += 1
                    
                if sum(fixture[i, j, s, t, r] for i in self.C for j in self.C for t in [5,6]) > 1: # 2+ Sunday games in stadium
                    violated += 1
                    critical += 1
                
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