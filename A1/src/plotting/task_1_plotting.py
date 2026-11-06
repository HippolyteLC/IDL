
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


LOG_DIR = Path(__file__).parent.parent.parent / "logs/task_1/final_runs"
FIG_PATH = "training_results.pdf"


def load_training_runs(log_directory):
    grouped_runs = defaultdict(list)
    configurations = {}
    log_directory = Path(log_directory)

    hp_files = sorted(p for p in log_directory.glob("*training_hyperparameters*") if p.is_file())
    print(f"Found {len(hp_files)} hyperparameter files.")

    for path in hp_files:
        match = re.search(r"(?P<date>.+)_seed(?P<seed>\d+)_training_hyperparameters", path.name)
        if not match:
            continue

        seed, run_date = int(match.group("seed")), match.group("date")

        with path.open(encoding="utf-8") as f:
            hp = json.load(f)

        model = hp.get("model", {}).get("model")
        if isinstance(model, str):
            model = model.lower()
        else:
            model = next((name for name in ("mlp", "cnn") if name in path.name.lower()), None)

        if model not in ("mlp", "cnn"):
            continue

        dataset = hp.get("data", {}).get("dataset_name")
        if not dataset:
            continue
        dataset = dataset.lower()

        config = json.loads(json.dumps(hp))
        config.get("data", {}).pop("dataset_name", None)
        config.get("training", {}).pop("seed", None)
        config.get("training", {}).pop("num_epochs", None)

        for key in ("H", "W", "C_i"):
            config.get("model", {}).pop(key, None)

        config_key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        config_id = hashlib.sha256(f"{model}:{config_key}".encode()).hexdigest()

        configurations[config_id] = {"model": model, "hyperparameters": config, "configuration_key": config_key}

        log_name = f"{run_date}_seed{seed}_training_logs"
        log_path = next(p for p in (log_directory / log_name, log_directory / f"{log_name}.json", log_directory / f"{log_name}.jsonl") if p.is_file())

        with log_path.open(encoding="utf-8") as f:
            epochs = [json.loads(line) for line in f if line.strip()] if log_path.suffix == ".jsonl" else json.load(f)

        if isinstance(epochs, dict):
            epochs = epochs.get("epochs", epochs.get("history"))

        grouped_runs[(model, config_id, dataset)].append({"seed": seed, "epochs": epochs})
        print(f"Loaded {model.upper()}, dataset={dataset}, seed={seed}, configuration={config_id[:8]}")

    print(f"\nDistinct configurations: {len(configurations)}")
    for model in ("mlp", "cnn"):
        n = sum(c["model"] == model for c in configurations.values())
        print(f"{model.upper()} configurations: {n}")

    return grouped_runs, configurations


def calculate_mean_and_std(runs, metric_name, common_epochs_only=True):
    curves = [[float(e[metric_name]) for e in r["epochs"]] for r in runs]
    lengths = [len(c) for c in curves]

    if len(set(lengths)) > 1:
        print(f"Warning: {metric_name} has different epoch counts: {sorted(lengths)}")

    n = min(lengths) if common_epochs_only else max(lengths)
    values = np.full((len(curves), n), np.nan)

    for i, curve in enumerate(curves):
        values[i, :min(len(curve), n)] = curve[:n]

    return np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=0)


def assign_model_labels(configurations):
    labels, mapping = {}, {}

    for model in ("mlp", "cnn"):
        configs = sorted(((k, v) for k, v in configurations.items() if v["model"] == model), key=lambda x: x[1]["configuration_key"])

        for i, (identifier, details) in enumerate(configs, 1):
            label = f"{model.upper()} {i}"
            labels[identifier] = label
            mapping[label] = {"model": model, "hyperparameters": details["hyperparameters"]}

    return labels, mapping


def save_training_results_csv(grouped_runs, configurations, model_labels, output_path):
    metrics = [f"{dataset}_{split}_{metric}" for dataset in ("fashionmnist", "cifar10") for metric in ("loss", "accuracy") for split in ("train", "test")]
    fieldnames = ["model"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
    rows = []

    configs = sorted(configurations.items(), key=lambda x: (0 if x[1]["model"] == "mlp" else 1, x[1]["configuration_key"]))

    for config_id, details in configs:
        model = details["model"]
        row = {"model": model_labels[config_id]}

        for dataset in ("fashionmnist", "cifar10"):
            runs = grouped_runs.get((model, config_id, dataset), [])

            for split in ("train", "test"):
                for metric in ("loss", "accuracy"):
                    col = f"{dataset}_{split}_{metric}"
                    vals = [float(r["epochs"][-1][f"{split}_{metric}"]) for r in runs]
                    vals = [v for v in vals if np.isfinite(v)]

                    if metric == "accuracy":
                        vals = [v * 100 for v in vals]

                    row[f"{col}_mean"] = float(np.mean(vals)) if vals else ""
                    row[f"{col}_std"] = float(np.std(vals, ddof=0)) if vals else ""

        rows.append(row)

    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def plot_training_results(log_directory, figure_path, common_epochs_only=True):
    grouped_runs, configurations = load_training_runs(log_directory)
    labels, mapping = assign_model_labels(configurations)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = figure_path.with_name("training_results_summary.csv")
    save_training_results_csv(grouped_runs, configurations, labels, csv_path)

    config_path = figure_path.with_name("training_results_configurations.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, sort_keys=True)
        f.write("\n")

    train_color, test_color = "#2878B5", "#D94841"
    row_definitions = [("fashionmnist", "loss", "Loss"), ("fashionmnist", "accuracy", "Accuracy (%)"), ("cifar10", "loss", "Loss"), ("cifar10", "accuracy", "Accuracy (%)")]

    for model in ("mlp", "cnn"):
        model_configs = sorted(((k, v) for k, v in configurations.items() if v["model"] == model), key=lambda x: x[1]["configuration_key"])

        if len(model_configs) != 3:
            raise ValueError(f"Expected three {model.upper()} configurations, found {len(model_configs)}.")

        fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.2), sharex=True, squeeze=False)
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.105, top=0.86, hspace=0.30, wspace=0.12)
        fig.align_ylabels(axes[:, 0])

        for col, (config_id, _) in enumerate(model_configs):
            axes[0, col].set_title(labels[config_id], fontsize=10, fontweight="semibold", pad=10)

        for row, (dataset, metric, metric_label) in enumerate(row_definitions):
            row_values = []

            for col, (config_id, _) in enumerate(model_configs):
                ax = axes[row, col]
                runs = grouped_runs.get((model, config_id, dataset), [])

                if not runs:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    continue

                for split, color in (("train", train_color), ("test", test_color)):
                    mean, std = calculate_mean_and_std(runs, f"{split}_{metric}", common_epochs_only)

                    if metric == "accuracy":
                        mean, std = mean * 100, std * 100

                    x = np.arange(1, len(mean) + 1)
                    ax.plot(x, mean, color=color, linewidth=1.25)
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.17, linewidth=0)
                    row_values.extend([mean - std, mean + std])

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=8, length=3, width=0.7, direction="out")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

                if metric == "accuracy":
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.2f}"))

                if col > 0:
                    ax.tick_params(axis="y", labelleft=False)

            if metric == "accuracy":
                ymin, ymax = 0, 100
                yticks = np.linspace(0, 100, 5)
            else:
                ymin = 0
                ymax = max((float(np.nanmax(v)) for v in row_values), default=1.0) * 1.08
                ymax = max(ymax, 0.1)
                locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
                yticks = locator.tick_values(ymin, ymax)
                yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]

            for col in range(3):
                axes[row, col].set_ylim(ymin, ymax)
                axes[row, col].set_yticks(yticks)

            axes[row, 0].set_ylabel(metric_label, fontsize=9, labelpad=7)

        for row in range(4):
            axes[row, 0].yaxis.set_label_coords(-0.30, 0.5)

        fig.canvas.draw()

        for dataset_label, first_row in (("FashionMNIST", 0), ("CIFAR-10", 2)):
            top = axes[first_row, 0].get_position()
            bottom = axes[first_row + 1, 0].get_position()
            fig.text(0.035, (top.y1 + bottom.y0) / 2, dataset_label, rotation=90, ha="center", va="center", fontsize=9, fontweight="semibold")

        for ax in axes[:-1, :].flat:
            ax.tick_params(axis="x", labelbottom=False)

        fig.text(0.575, 0.040, "Number of Epochs", ha="center", va="center", fontsize=10)
        fig.legend(handles=[plt.Line2D([0], [0], color=train_color, linewidth=1.5, label="Train"), plt.Line2D([0], [0], color=test_color, linewidth=1.5, label="Test")], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=9)

        output_path = figure_path.with_name(f"training_results_{model}")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {output_path.with_suffix('.pdf')}")
        print(f"Saved {output_path.with_suffix('.png')}")

    print(f"Saved configuration mapping: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MLP and CNN training results and export a CSV summary.")
    parser.add_argument("--log-directory", type=Path, default=LOG_DIR)
    parser.add_argument("--figure-path", type=Path, default=FIG_PATH)
    parser.add_argument("--all-epochs", action="store_true", help="Keep longer runs instead of truncating to the shortest run.")
    args = parser.parse_args()

    plot_training_results(args.log_directory, args.figure_path, common_epochs_only=not args.all_epochs)


if __name__ == "__main__":
    main()
