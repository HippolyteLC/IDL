
#!/bin/bash

# Stop if a command fails
set -e

# Loop through seeds and datasets
for seed in 111 222 333 456 567 ; do
    for dataset in fashionmnist cifar10; do
        echo "=========================================="
        echo "Running MLP 1: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model mlp \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 128 \
            --learning-rate 0.003 \
            --activation-method leakyrelu \
            --n-hidden-fc-layers 7 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --test

        echo ""

        echo "=========================================="
        echo "Running MLP 2: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model mlp \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 256 \
            --learning-rate 0.01 \
            --activation-method relu \
            --n-hidden-fc-layers 7 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --test

        echo ""

        echo "=========================================="
        echo "Running MLP 3: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model mlp \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 128 \
            --learning-rate 0.003 \
            --activation-method leakyrelu \
            --n-hidden-fc-layers 8 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --test

        echo ""
    done
done


echo "All MLP test runs completed!"

# Loop through seeds and datasets for CNN
for seed in 111 222 333 456 567; do
    for dataset in fashionmnist cifar10; do
        echo "=========================================="
        echo "Running CNN 1: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model cnn \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 128 \
            --learning-rate 0.003 \
            --activation-method relu \
            --batch-norm \
            --n-hidden-fc-layers 3 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --kernel-size 3 \
            --test

        echo ""

        echo "=========================================="
        echo "Running CNN 2: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model cnn \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 256 \
            --learning-rate 0.003 \
            --activation-method relu \
            --batch-norm \
            --n-hidden-fc-layers 2 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --kernel-size 7 \
            --test

        echo ""

        echo "=========================================="
        echo "Running CNN 3: dataset=$dataset, seed=$seed"
        echo "=========================================="

        python3 -m src.scripts.experiment \
            --model cnn \
            --num-epochs 20 \
            --dataset "$dataset" \
            --fc-width 256 \
            --learning-rate 0.01 \
            --activation-method relu \
            --batch-norm \
            --n-hidden-fc-layers 2 \
            --split-seed 2026 \
            --seed "$seed" \
            --dropout 0.2 \
            --batch-size 64 \
            --kernel-size 3 \
            --test

        echo ""
    done
done

echo "All CNN test runs completed!"
