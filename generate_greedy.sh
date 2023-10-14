seeds=("1" "2")
years=("2022" "2023")

for seed in "${seeds[@]}"; do
    for year in "${years[@]}"; do
        python random_greedy_fixtures1.py $seed $year > "solutions/greedy/${year}-greedy1-${seed}.txt" &
        python random_greedy_fixtures2.py $seed $year > "solutions/greedy/${year}-greedy2-${seed}.txt" &
    done
done