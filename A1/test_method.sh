#!/bin/bash

# Loop through methods 1 to 7
for method in {1..7}; do
    echo "=========================================="
    echo "Running test for method: $method"
    echo "=========================================="
    
    python3 A1/code/experiment.py \
    --model clock_cnn \
    --num-epochs 20 \
    --dataset clock_75 \
    --clock-method $method \
    --fc-width 4096 
    echo ""
done

echo "All test runs completed!"