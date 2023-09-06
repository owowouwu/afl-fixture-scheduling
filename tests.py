from schedule import Team, Stadium, Tournament
import unittest
import numpy as np

class TestTeam(unittest.TestCase):

    def test_team_creation(self):
        team = Team("Team A", 90, Stadium("Stadium A", 1.0, 2.0, "City A"))
        self.assertEqual(team.name, "Team A")
        self.assertEqual(team.rating, 90)
        self.assertEqual(team.home.name, "Stadium A")

class TestStadium(unittest.TestCase):

    def test_stadium_creation(self):
        stadium = Stadium("Stadium X", 3.0, 4.0, "City X")
        self.assertEqual(stadium.name, "Stadium X")
        self.assertEqual(stadium.x, 3.0)
        self.assertEqual(stadium.y, 4.0)
        self.assertEqual(stadium.geolocation, "City X")

class TestTournament(unittest.TestCase):

    def setUp(self):
        self.stadium1 = Stadium("Stadium A", 1.0, 2.0, "City A")
        self.stadium2 = Stadium("Stadium B", 3.0, 4.0, "City B")
        self.stadium3 = Stadium("Stadium C", 3.0, 4.0, "City C")
        self.team1 = Team("Team 1", 80, self.stadium1)
        self.team2 = Team("Team 2", 85, self.stadium2)
        self.team3 = Team("Team 3", 90, self.stadium3)
        self.tournament = Tournament(5, 3, [self.stadium1, self.stadium2, self.stadium3],
         [self.team1, self.team2, self.team3])
        s = np.random.randint(3,size=(5, 3, 3))
        print(s)
        self.tournament.schedule = s

    def test_tournament_creation(self):
        self.assertEqual(self.tournament.rounds, 5)
        self.assertEqual(self.tournament.timeslots, 3)
        self.assertEqual(len(self.tournament.stadiums), 3)
        self.assertEqual(len(self.tournament.teams), 3)

    # def test_schedule_property(self):
    #     # Test schedule property getter and setter
    #     schedule = np.random.randint(1,size=(5, 3, 3))  # Example schedule
    #     self.tournament.schedule = schedule
    #     self.assertTrue(np.array_equal(self.tournament.schedule, schedule))

    def test_schedule_tsswap(self):
        t1 = 0
        t2 = 1
        print(f"testing team swap of teams {t1} and {t2}")
        # schedule = np.random.randint(1,size=(5, 3, 3))
        # self.tournament.schedule = schedule
        print("current schedule\n", self.tournament.schedule)
        print("new schedule\n", self.tournament.team_swap_all(t1,t2))

    def test_schedule_homeawayswap(self):
        round = 2
        slot = 1
        print(f"testing homeaway swap for round {round} timeslot {slot}")
        # schedule = np.random.randint(1,size=(5, 3, 3))
        # self.tournament.schedule = schedule
        print("current schedule\n", self.tournament.schedule)
        print("new schedule\n", self.tournament.home_away_swap(round=round, slot=slot))

    def test_schedule_swaptimeslot(self):
        s1 = (2,1)
        s2 = (1,1)
        print(f"testing swap of rounds {s1} and {s2}")
        # schedule = np.random.randint(1,size=(5, 3, 3))
        # self.tournament.schedule = schedule
        print("current schedule\n", self.tournament.schedule)
        print("new schedule\n", self.tournament.swap_timeslot(r1=s1[0], r2=s2[0], ts1=s1[1],ts2=s2[1]))

if __name__ == '__main__':
    unittest.main()