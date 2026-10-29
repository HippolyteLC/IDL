
#!/bin/bash

# Loop through seeds
for seed in 111 222 333 456 567; do
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
