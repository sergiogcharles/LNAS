# exps=(0 1 2 3 4 5)
# sparsity=(0.167 0.333 0.499 0.666 0.833 0.950)
#exps=(0 1 2 3 4 5 6 7)
#sparsity=(0.125 0.25 0.375 0.50 0.625 0.75 0.875 0.950)
exps=(0)
sparsity=(0.125)
for exp in "${exps[@]}"; do
    echo "-------------------Running exp $exp------------------------"
    python3 retrain_lottery.py --sparsity ${sparsity[exp]} --exp $exp
done
