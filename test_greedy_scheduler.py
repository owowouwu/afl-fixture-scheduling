import numpy as np
import json
from modules.tournament import Tournament
from modules.greedy_scheduler import GreedyScheduler
def main(seed, rcl_length):
    with open('teams.json', 'r') as f:
        teams = json.load(f)

    with open('locations.json', 'r') as f:
        locations = json.load(f)

    stadiums = {}
    for loc in locations:
        snames = []
        for s in locations[loc]['stadiums']:
            snames.append(s)
            stadiums[s] = {
                'location': loc,
                'size': locations[loc]['stadiums'][s]['size']
            }

        locations[loc]['stadiums'] = snames

    timeslot_values = [10, 13, 6, 7, 11, 5, 4]  # Change later according to attendances
    timeslot_names = ['Thursday Night', 'Friday Night', 'Saturday Afternoon', 'Saturday Evening',
                      'Saturday Night', 'Sunday Afternoon', 'Sunday Evening']

    timeslots = [{'value': v, 'name': n} for (n, v) in zip(timeslot_names, timeslot_values)]
    tourn = Tournament(teams=teams, locations=locations, stadiums=stadiums, timeslots=timeslots, rounds=22)

    scheduler = GreedyScheduler(tourn)

    fixture, timeslot_usage, stadium_usage = scheduler.generate_fixture_from_matchlist(rcl_length=rcl_length, seed=seed, print_fixture=True)
    tourn.fixture_matrix = fixture
    np.save('output/greedy.npy', fixture)
    print("objective: ", tourn.fixture_attractiveness(fixture=fixture))
    print("constraints violated: ", tourn.feasibility(fixture, debug=True))

if __name__ == '__main__':
    rcl_length = 10
    seed = 1234
    main(seed, rcl_length)