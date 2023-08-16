# Brainstorm

## Parameters 

- a graph of locations (cities). In each location we have a number of playing grounds.
- a number of teams T. Each team is associated with a particular home location and may also have a particular playing ground.
	- team parameters
		- membership
		- rating
		- popularity
- a game is played between two teams at a particular stadium.
	- a game's viewership and attendance can be affected
		- by the day, 
		- time of day, 
		- stadium 
		- teams playing

## (Possible) Objectives

- maximise the 'value' of matches over the season
	- can be measured by a number of ways - 
		- viewership/attendance, 
		- how good each team is and how close the teams are in rank
			- are there any rivalries?

objective function can be a combination of the things

## Decision Variables

scheduling the time and location of the games each team is going to play in the season

## Constraints

- team travel restrictions
- must be time between games for a team
	- can be a constraint or added as a penalty
- precedence - teams may have 
- bespoke constraints - purposefully scheduling games between rivals
- minimise the clashes between 'good' games that may occur concurrently
- minimise consecutive byes
- teams can also be required to travel to specific locations to promote less popular regions.

# Specific Sports

- AFL
	- 18 team round robin with 5 additional rounds 
- World Cup
	- 32 teams, 8 houses, round robin in house
