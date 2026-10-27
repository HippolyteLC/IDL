import json
from pathlib import Path

directory = Path("path/to/dir")

for params_file in directory.glob("*_training_hyperparameters.json"):
    run_id = params_file.name.removesuffix("_training_hyperparameters.json")
    logs_file = directory / f"{run_id}_training_logs.jsonl"

    if not logs_file.exists():
        continue

    try:
        params = json.loads(params_file.read_text())
        if "last_train_accuracy" in params and "last_val_accuracy" in params:
            continue

        lines = logs_file.read_text().splitlines()
        last = json.loads(next(line for line in reversed(lines) if line.strip()))

        params["last_train_accuracy"] = last["train_accuracy"]
        params["last_val_accuracy"] = last["validation_accuracy"]

        params_file.write_text(json.dumps(params, indent=4) + "\n")
    except (json.JSONDecodeError, KeyError, StopIteration) as e:
        print(f"Skipping {params_file.name if not isinstance(e, json.JSONDecodeError) else params_file.name}: {e}")