
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


LOG_DIR = Path(__file__).parent
FIG_PATH = Path("clock_methods.pdf")
CSV_NAME = "clock_results.csv"

colors = {
    1: "#D9D9D9",
    2: "#A6A6A6",
    3: "#0072B2",
    4: "#E69F00",
    5: "#737373",
    6: "#404040",
    7: "#009E73",
}


def load_all_runs(log_dir, dataset=None):
    runs = defaultdict(list)
    configs = defaultdict(set)

    for hp_file in sorted(Path(log_dir).glob("*training_hyperparameters*")):
        match = re.search(r"(.+)_seed(\d+)_training_hyperparameters", hp_file.name)
        if not match:
            continue

        date, seed = match.group(1), int(match.group(2))

        with open(hp_file, encoding="utf-8") as f:
            hp = json.load(f)

        method = hp["model"]["clock_method"]

        if dataset and hp["data"]["dataset_name"].lower() != dataset.lower():
            continue

        log_name = f"{date}_seed{seed}_training_logs"
        log_file = next(
            p for p in (
                Path(log_dir) / log_name,
                Path(log_dir) / f"{log_name}.json",
                Path(log_dir) / f"{log_name}.jsonl"
            ) if p.is_file()
        )

        with open(log_file, encoding="utf-8") as f:
            if log_file.suffix == ".jsonl":
                rec = [json.loads(line) for line in f if line.strip()]
            else:
                rec = json.load(f)

        if isinstance(rec, dict):
            rec = rec.get("epochs", rec.get("history"))

        # ignore the stuff that is expected to change between runs
        cfg = json.loads(json.dumps(hp))
        cfg["training"].pop("seed", None)
        cfg["training"].pop("num_epochs", None)
        cfg["model"].pop("out_feat", None)
        cfg["model"].pop("num_output_heads", None)

        configs[method].add(json.dumps(cfg, sort_keys=True))
        runs[method].append({"seed": seed, "epochs": rec})

        print(f"Loaded method {method}, seed {seed}, {len(rec)} epochs")

    for m, c in configs.items():
        if len(c) > 1:
            print(f"Warning: different configs found for method {m}")

    return runs


def mean_and_std(runs, metric, common=True):
    vals = [[float(e[metric]) for e in r["epochs"]] for r in runs]
    lengths = [len(v) for v in vals]
    n = min(lengths) if common else max(lengths)

    if len(set(lengths)) > 1:
        print(f"Different epoch counts for {metric}: {sorted(lengths)}")

    arr = np.full((len(vals), n), np.nan)

    for i, v in enumerate(vals):
        arr[i, :min(len(v), n)] = v[:n]

    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def plot_clock_results(log_dir, fig_path, methods=range(1, 8),
                       dataset=None, common=True):

    runs = load_all_runs(log_dir, dataset)
    methods = list(methods)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.15,
                        top=0.80, wspace=0.25)

    for ax, metric, title in zip(
        axes,
        ("loss", "accuracy"),
        ("Test Loss", "Test Error (minutes)")
    ):
        ends = []

        for m in methods:
            rr = runs.get(m, [])
            if not rr:
                continue

            c = colors.get(m, "#000000")
            y, std = mean_and_std(rr, f"test_{metric}", common)
            x = np.arange(1, len(y) + 1)

            ax.plot(x, y, color=c, linewidth=1.8)
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.15)

            ends.append((m, x[-1], y[-1], c))

        # labels need a little room between them
        if ends:
            ends.sort(key=lambda z: z[2])
            ymin, ymax = ax.get_ylim()
            gap = (ymax - ymin) * 0.055
            label_y = []

            for m, x, y, c in ends:
                label_y.append(max(y, label_y[-1] + gap) if label_y else y)

            overflow = label_y[-1] - (ymax - (ymax - ymin) * 0.02)
            if overflow > 0:
                label_y = [v - overflow for v in label_y]

            for (m, x, y, c), yy in zip(ends, label_y):
                ax.annotate(
                    f"Method {m}",
                    xy=(x, y),
                    xytext=(x + 0.03 * max(x, 1), yy),
                    color=c,
                    fontsize=8,
                    fontweight="semibold",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="-", color=c,
                        linewidth=0.7, alpha=0.7
                    ),
                    annotation_clip=False
                )

        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Number of Epochs", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=3, width=0.7)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.35)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots to {fig_path}")


def export_clock_table(log_dir, csv_path, methods=range(1, 8),
                       dataset=None, common=True, epoch=None):

    runs = load_all_runs(log_dir, dataset)
    metrics = ("train_loss", "train_accuracy", "test_loss", "test_accuracy")

    rows = []

    for m in methods:
        rr = runs.get(m, [])
        if not rr:
            continue

        row = {"clock_method": m, "n_seeds": len(rr)}

        for metric in metrics:
            y, std = mean_and_std(rr, metric, common)
            idx = len(y) - 1 if epoch is None else epoch - 1

            row["epoch"] = idx + 1
            row[f"{metric}_mean"] = y[idx]
            row[f"{metric}_std"] = std[idx]

        rows.append(row)

    header = ["clock_method", "n_seeds", "epoch"]
    for metric in metrics:
        header.extend([f"{metric}_mean", f"{metric}_std"])

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                k: f"{v:.6g}" if isinstance(v, float) else v
                for k, v in r.items()
            })

    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot traning results for different clock methods"
    )

    parser.add_argument("--log-directory", type=Path, default=LOG_DIR)
    parser.add_argument("--figure-path", type=Path, default=FIG_PATH)
    parser.add_argument("--methods", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--table-epoch", type=int, default=None)
    parser.add_argument("--all-epochs", action="store_true")

    args = parser.parse_args()
    common = not args.all_epochs

    plot_clock_results(
        args.log_directory, args.figure_path,
        args.methods, args.dataset, common
    )

    if not args.no_csv:
        csv_path = args.csv_path or args.figure_path.parent / CSV_NAME

        export_clock_table(
            args.log_directory, csv_path,
            args.methods, args.dataset,
            common, args.table_epoch
        )


if __name__ == "__main__":
    main()
