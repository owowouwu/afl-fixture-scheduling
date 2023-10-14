seeds=("1" "2")
years=("2022" "2023")

for seed in "${seeds[@]}"; do
    for year in "${years[@]}"; do
        python grasp.py $seed $year "by_round" > "solutions/grasp/${year}grasp1-${seed}.txt" &
        python grasp.py $seed $year "whole_fixture" > "solutions/grasp/${year}grasp2-${seed}.txt" &
    done
done