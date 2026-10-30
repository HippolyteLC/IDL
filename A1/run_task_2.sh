
#!/bin/bash
# 111 222 333 456 567
# Loop through seeds
<<<<<<< HEAD
for seed in 456 567; do
=======
for seed in 567 ; do
>>>>>>> 414e4d82a9ce5c77fe08c056467a50dd56cd6f75
    # Loop through methods 1 to 7
    for method in {1..7}; do
        echo "=========================================="
        echo "Running test for method: $method, seed $seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model clock_cnn \
            --num-epochs 20 \
            --dataset clock_150 \
            --clock-method "$method" \
            --fc-width 256 \
            --batch-norm \
            --learning-rate 0.0001 \
            --n-hidden-fc-layers 2 \
            --split-seed 2026 \
            --seed "$seed" \
            --batch-size 32 \
            --test

        echo ""
    done
done

echo "All test runs completed!"
