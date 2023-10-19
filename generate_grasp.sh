seeds=("1" "2")
years=("2022" "2023")

for seed in "${seeds[@]}"; do
    for year in "${years[@]}"; do
        python grasp.py $seed $year "by_round" > "solutions/grasp/${year}grasp1-${seed}.txt" &
    done
done
