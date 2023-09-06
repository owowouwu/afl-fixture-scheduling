import numpy as np

class Team:

    def __init__(self, name, rating, home, rivals = None):
        self.name = name
        self.rating = rating
        self.home = home
        self.rivals = rivals
        self.homeidx = None

class Stadium:

    def __init__(self, name:str, x:float, y:float, geolocation:str):
        self.name = name
        self.x = x
        self.y = y
        # should correspond to like a city or something, travel within the same city is not as penalised
        self.geolocation = geolocation

class Tournament:

    def __init__(self, rounds:int, timeslots: int, stadiums: list[Stadium],
                teams: list[Team]):

        self.teams = teams
        self.n_teams = len(teams)
        self.rounds = rounds
        self.timeslots = timeslots
        self.stadiums = stadiums
        for t in teams:
            t.homeidx = stadiums.index(t.home)
        self.locations, self.d_matrix = self._init_locations(stadiums)
        self._schedule = None

    def _init_locations(self,stadiums : list[Stadium]):
        res = {}
        for s in stadiums:
            if s.geolocation not in res:
                res[s.geolocation] = (s.x, s.y)

        locations = list(res.keys())
        d_matrix = []
        for i, l1 in enumerate(locations):
            d_matrix.append([])
            for j,l2 in enumerate(locations):
                d_matrix[i].append(np.sqrt(
                    (res[l1][0] - res[l2][0])**2 + (res[l1][1] - res[l2][1])**2
                ))

        return locations, d_matrix

    @property
    def schedule(self) -> np.ndarray[np.int64]:
        return self._schedule

    @schedule.setter
    def schedule(self, schedule: np.ndarray[np.int64]):
        # verify that dimensions are correct
        dims = np.shape(schedule)
        if len(dims) != 3:
            raise ValueError("a schedule should be a 3 dimensional matrix.")

        if (dims[0] != self.rounds) or (dims[1] != self.timeslots) or (dims[2] != 3):
            raise ValueError("dimension mismatch. a schedule should be a rounds x timeslots x 3 matrix")
        
        # check valid home and away arrangements
        # for i,round in enumerate(schedule):
        #     for j,timeslot in enumerate(round):
        #         if timeslot[2] != self.teams[timeslot[0]].homeidx:
        #             raise ValueError(f"game {j} in round {i} does not have a valid stadium.")

        self._schedule = schedule

    
    def check_feasibility(self, schedule) -> bool:
        
        pass

    def objective(self, schedule) -> float:

        pass
             
    def home_away_swap(self, round:int, slot:int, schedule = None):
        # swap home and away
        new_schedule = schedule if schedule is not None else self._schedule.copy()
    
        new_schedule[round][slot][0], new_schedule[round][slot][1] = (
            new_schedule[round][slot][1], new_schedule[round][slot][0]
        )
        # adjust stadium
        new_home_team = new_schedule[round][slot][0]
        new_schedule[round][slot][2] = self.teams[new_home_team].homeidx

        return new_schedule

    def swap_timeslot(self, r1:int, r2:int, ts1:int, ts2:int, schedule = None):
        new_schedule = schedule if schedule is not None else self._schedule.copy()
        new_schedule[r1][ts1], new_schedule[r2][ts2] = new_schedule[r2][ts2], new_schedule[ts1][r1]

        return new_schedule


    def team_swap_round(self, t1:int, t2:int, round:int, schedule = None):
        new_schedule = schedule if schedule is not None else self._schedule.copy()
        for timeslot in enumerate(new_schedule[j]):
            match1 = np.where(timeslot == t1)
            match2 = np.where(timeslot == t2)

            # if found t1 replace with t2
            if match1.size > 0:
                slot = match1[0]
                is_home = (slot == 0)
                timeslot[slot] = t2
                if is_home: timeslot[2] = self.teams[t2].homeidx
            
            # if found t2 replace with t1
            if match2.size > 0:
                slot = match2[0]
                is_home = (slot == 0)
                timeslot[slot] = t1
                if is_home: timeslot[2] = self.teams[t1].homeidx

        return new_schedule
    
    def team_swap_all(self, t1, t2, schedule = None):
        new_schedule = schedule if schedule is not None else self._schedule.copy()
        for i, round in enumerate(new_schedule):
            for j, timeslot in enumerate(round):
                match1 = np.where(timeslot == t1)[0]
                match2 = np.where(timeslot == t2)[0]

                # if found t1 replace with t2
                if match1.size > 0:
                    slot = match1[0]
                    is_home = (slot == 0)
                    timeslot[slot] = t2
                    if is_home: timeslot[2] = self.teams[t2].homeidx
                
                # if found t2 replace with t1
                if match2.size > 0:
                    slot = match2[0]
                    is_home = (slot == 0)
                    timeslot[slot] = t1
                    if is_home: timeslot[2] = self.teams[t1].homeidx

        return new_schedule
