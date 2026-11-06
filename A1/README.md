## Experiments

1. to load necessary modules use the following command on the LLW machines. 
```bash
module load LIACS/ItDL-2026
```
2. run a test on clock dataset
```bash
cd A1

python3 -m src.scripts.experiment --model clock_cnn --num-epochs 1 --dataset clock_75 --clock-method 1
```

3. run experiments for task 1 (linux)
```bash
chmod +x ./run_task_1.sh

./run_task_1.sh
```

4. run experiments for task 2 (linux)
```bash
chmod +x ./run_task_2.sh

./run_task_2.sh
```

## Results 

Results are logged separately for task 1 and 2. All results for task 1 can be found inside `logs/task_1`, for task 2 inside `logs/task_2`. 
There is a subfolder for the results of each method: *mlp_None, cnn_None* for task 1, *clock_cnn_1, clock_cnn_2, ..., clock_cnn_7* for task 2. 

Model, training and experiment parameters are stored inside json files. Train, val, and test loss and accuracies are stored inside jsonl files. 


