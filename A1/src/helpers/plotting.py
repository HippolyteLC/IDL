import json
import matplotlib.pyplot as plt


def plot_metrics(jsonl_file):
    with open(jsonl_file) as f:
        data = [json.loads(line) for line in f if line.strip()]

    for metric in ["loss", "accuracy"]:
        plt.figure()
        plt.plot([d[f"train_{metric}"] for d in data], label="Train")
        plt.plot([d[f"validation_{metric}"] for d in data], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Train vs. Validation {metric.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"metrics_{metric}.png")

plot_metrics("A1/logs/20261020_175945_training_logs.jsonl")


