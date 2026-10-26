1. to load necessary modules use the following command on the LLW machines. 
```bash
module load LIACS/ItDL-2026
```
2. run a test on clock dataset
```bash
python3 A1/code/experiment.py --model clock_cnn --num-epochs 2 --dataset clock_75 --clock-method 2
```