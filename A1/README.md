1. to load necessary modules use the following command on the LLW machines. 
```bash
module load LIACS/ItDL-2026
```
2. run a test on clock dataset
```bash
cd A1

python3 -m src.scripts.experiment --model clock_cnn --num-epochs 1 --dataset clock_75 --clock-method 1
```

# TODO:
1. fix multi output head classifier (dims), method 3.

2. fix the loss stagnation task 2, all methods (except for 1).