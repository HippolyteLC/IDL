
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path(__file__).parent.parent.parent / "logs/task_1"
FIG_PATH = "task_1_aggr_results.pdf"

def load_training_runs(log_directory):
    """Discover training runs and group them by model, configuration, and dataset."""
    grouped_runs = defaultdict(list)
    configurations = {}

    log_directory = Path(log_directory)

    if not log_directory.is_dir():
        raise FileNotFoundError(
            f"Log directory does not exist: {log_directory.resolve()}"
        )

    hyperparameter_paths = sorted(
        path
        for path in log_directory.rglob("*training_hyperparameters*")
        if path.is_file()
    )

    if not hyperparameter_paths:
        raise FileNotFoundError(
            f"No hyperparameter files found under "
            f"{log_directory.resolve()}. Expected filenames containing "
            f"'training_hyperparameters'."
        )

    print(f"Found {len(hyperparameter_paths)} hyperparameter files.")

    for hyperparameter_path in hyperparameter_paths:
        filename_match = re.search(
            r"(?P<date>.+)_seed(?P<seed>\d+)_training_hyperparameters",
            hyperparameter_path.name,
        )

        if filename_match is None:
            print(
                f"Skipping unrecognized filename: {hyperparameter_path}"
            )
            continue

        seed = int(filename_match.group("seed"))
        run_date = filename_match.group("date")

        try:
            with hyperparameter_path.open(encoding="utf-8") as file:
                hyperparameters = json.load(file)
        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipping unreadable file {hyperparameter_path}: {error}")
            continue

        model_name = (
            hyperparameters.get("model", {}).get("model")
            or next(
                (
                    parent.name.removesuffix("_None")
                    for parent in hyperparameter_path.parents
                    if parent.name in ("mlp_None", "cnn_None")
                ),
                None,
            )
        )

        if model_name not in ("mlp", "cnn"):
            print(
                f"Skipping file with unknown model '{model_name}': "
                f"{hyperparameter_path}"
            )
            continue

        dataset_name = (
            hyperparameters.get("data", {}).get("dataset_name")
        )

        if not dataset_name:
            print(
                f"Skipping file without data.dataset_name: "
                f"{hyperparameter_path}"
            )
            continue

        dataset_name = dataset_name.lower()


        configuration = json.loads(json.dumps(hyperparameters))

        configuration.get("data", {}).pop("dataset_name", None)
        configuration.get("training", {}).pop("seed", None)

        model_configuration = configuration.get("model", {})
        model_configuration.pop("H", None)
        model_configuration.pop("W", None)
        model_configuration.pop("C_i", None)

        configuration_key = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        )

        configuration_identifier = hashlib.sha256(
            f"{model_name}:{configuration_key}".encode("utf-8")
        ).hexdigest()

        configurations[configuration_identifier] = {
            "model": model_name,
            "hyperparameters": configuration,
            "configuration_key": configuration_key,
        }


        log_filename = f"{run_date}_seed{seed}_training_logs"

        possible_log_paths = [
            hyperparameter_path.parent / log_filename,
            hyperparameter_path.parent / f"{log_filename}.json",
            hyperparameter_path.parent / f"{log_filename}.jsonl",
        ]

        training_log_path = next(
            (
                path
                for path in possible_log_paths
                if path.is_file()
            ),
            None,
        )

        if training_log_path is None:
            matching_log_paths = [
                path
                for path in log_directory.rglob(f"{log_filename}*")
                if path.is_file()
                and path.name in (
                    log_filename,
                    f"{log_filename}.json",
                    f"{log_filename}.jsonl",
                )
            ]

            if len(matching_log_paths) != 1:
                print(
                    f"Skipping run: expected one matching log for "
                    f"{hyperparameter_path.name}, found "
                    f"{len(matching_log_paths)}."
                )
                continue

            training_log_path = matching_log_paths[0]

        try:
            with training_log_path.open(encoding="utf-8") as file:
                if training_log_path.suffix == ".jsonl":
                    epoch_records = [
                        json.loads(line)
                        for line in file
                        if line.strip()
                    ]
                else:
                    epoch_records = json.load(file)

            if isinstance(epoch_records, dict):
                epoch_records = epoch_records.get(
                    "epochs",
                    epoch_records.get("history"),
                )

            if not isinstance(epoch_records, list) or not epoch_records:
                raise ValueError(
                    "Expected a non-empty list of epoch records."
                )

            required_metrics = {
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            }

            if not all(
                isinstance(record, dict)
                and required_metrics.issubset(record)
                for record in epoch_records
            ):
                raise ValueError(
                    "Epoch records are missing one or more required metrics."
                )

        except (json.JSONDecodeError, OSError, ValueError) as error:
            print(f"Skipping invalid log {training_log_path}: {error}")
            continue

        grouped_runs[
            (model_name, configuration_identifier, dataset_name)
        ].append(
            {
                "seed": seed,
                "epochs": epoch_records,
            }
        )

        print(
            f"Loaded {model_name.upper()}, dataset={dataset_name}, "
            f"seed={seed}, configuration={configuration_identifier[:8]}"
        )

    print(f"\nDistinct configurations discovered: {len(configurations)}")

    for model_name in ("mlp", "cnn"):
        configuration_count = sum(
            details["model"] == model_name
            for details in configurations.values()
        )
        print(
            f"{model_name.upper()} configurations: {configuration_count}"
        )

    if not configurations:
        raise ValueError(
            f"No valid configurations were loaded from "
            f"{log_directory.resolve()}. Check the printed skip messages, "
            f"the filename patterns, and the JSON structure."
        )

    return grouped_runs, configurations


def calculate_mean_and_standard_deviation(runs, metric_name):
    """Calculate per-epoch mean and standard deviation across seeds."""
    seed_curves = []

    for run in runs:
        seed_curves.append(
            [
                float(epoch_record[metric_name])
                for epoch_record in run["epochs"]
            ]
        )

    if not seed_curves:
        raise ValueError(f"No runs found for {metric_name}.")

    epoch_counts = {len(curve) for curve in seed_curves}

    if len(epoch_counts) != 1:
        raise ValueError(
            f"Inconsistent epoch counts for {metric_name}: "
            f"{sorted(epoch_counts)}"
        )

    seed_curves = np.asarray(seed_curves, dtype=float)

    return (
        np.mean(seed_curves, axis=0),
        np.std(seed_curves, axis=0, ddof=0),
    )


def assign_model_labels(configurations):
    """Assign stable row labels to distinct model configurations."""
    model_labels = {}
    configuration_mapping = {}

    for model_name in ("mlp", "cnn"):
        model_configurations = sorted(
            [
                (identifier, details)
                for identifier, details in configurations.items()
                if details["model"] == model_name
            ],
            key=lambda item: item[1]["configuration_key"],
        )

        for configuration_index, (
            configuration_identifier,
            details,
        ) in enumerate(model_configurations, start=1):
            model_label = f"{model_name.upper()} {configuration_index}"

            model_labels[configuration_identifier] = model_label
            configuration_mapping[model_label] = {
                "model": model_name,
                "hyperparameters": details["hyperparameters"],
            }

    return model_labels, configuration_mapping




def plot_training_results(log_directory, figure_path):
    """Create separate,  MLP and CNN figures."""
    grouped_runs, configurations = load_training_runs(log_directory)
    model_labels, configuration_mapping = assign_model_labels(configurations)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    configuration_path = figure_path.with_name(
        f"{figure_path.stem}_configurations.json"
    )

    with configuration_path.open("w", encoding="utf-8") as file:
        json.dump(configuration_mapping, file, indent=4, sort_keys=True)
        file.write("\n")

    train_color = "#2878B5"
    test_color = "#D94841"

    row_definitions = [
        ("fashionmnist", "loss", "Loss"),
        ("fashionmnist", "accuracy", "Accuracy (%)"),
        ("cifar10", "loss", "Loss"),
        ("cifar10", "accuracy", "Accuracy (%)"),
    ]

    for model_name in ("mlp", "cnn"):
        model_configurations = sorted(
            [
                (identifier, details)
                for identifier, details in configurations.items()
                if details["model"] == model_name
            ],
            key=lambda item: item[1]["configuration_key"],
        )

        if len(model_configurations) != 3:
            raise ValueError(
                f"Expected three {model_name.upper()} configurations, "
                f"but found {len(model_configurations)}."
            )

        figure, axes = plt.subplots(
            nrows=4,
            ncols=3,
            figsize=(7.2, 7.5),
            sharex=True,
            squeeze=False,
        )

        figure.subplots_adjust(
            left=0.17,
            right=0.98,
            bottom=0.12,
            top=0.86,
            hspace=0.30,
            wspace=0.12,
        )
        figure.align_ylabels(axes[:, 0])

        for column_index, (configuration_identifier, _) in enumerate(
            model_configurations
        ):
            axes[0, column_index].set_title(
                model_labels[configuration_identifier],
                fontsize=10,
                fontweight="semibold",
                pad=10,
            )

        for row_index, (
            dataset_name,
            metric_name,
            metric_label,
        ) in enumerate(row_definitions):
            row_values = []

            for column_index, (configuration_identifier, _) in enumerate(
                model_configurations
            ):
                axis = axes[row_index, column_index]

                runs = grouped_runs.get(
                    (
                        model_name,
                        configuration_identifier,
                        dataset_name,
                    ),
                    [],
                )

                if not runs:
                    axis.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
                    continue

                for split_name, color in (
                    ("train", train_color),
                    ("test", test_color),
                ):
                    
                    mean_values, standard_deviation = (
                        calculate_mean_and_standard_deviation(
                            runs,
                            f"{split_name}_{metric_name}",
                        )
                    )

                    if metric_name == "accuracy":
                        mean_values = mean_values * 100
                        standard_deviation = standard_deviation * 100

                    epoch_numbers = np.arange(1, len(mean_values) + 1)

                    axis.plot(
                        epoch_numbers,
                        mean_values,
                        color=color,
                        linewidth=1.25,
                    )

                    axis.fill_between(
                        epoch_numbers,
                        mean_values - standard_deviation,
                        mean_values + standard_deviation,
                        color=color,
                        alpha=0.17,
                        linewidth=0,
                    )

                    row_values.extend(
                        [
                            mean_values - standard_deviation,
                            mean_values + standard_deviation,
                        ]
                    )


                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.grid(False)

                axis.tick_params(
                    axis="both",
                    which="major",
                    labelsize=8,
                    length=3,
                    width=0.7,
                    direction="out",
                )

                if metric_name == "accuracy":
                    axis.yaxis.set_major_formatter(
                        plt.FuncFormatter(
                            lambda value, _: f"{value:.2f}"
                        )
                    )

                if column_index > 0:
                    axis.tick_params(axis="y", labelleft=False)
                    axis.set_ylabel("")

            if row_values:
                row_minimum = min(
                    float(np.nanmin(values))
                    for values in row_values
                )
                row_maximum = max(
                    float(np.nanmax(values))
                    for values in row_values
                )

                row_range = row_maximum - row_minimum

                if row_range == 0:
                    row_range = max(abs(row_maximum) * 0.1, 0.01)

                padding = row_range * 0.08
                shared_minimum = row_minimum - padding
                shared_maximum = row_maximum + padding

                shared_ticks = np.linspace(
                    shared_minimum,
                    shared_maximum,
                    4,
                )

                for column_index in range(3):
                    axes[row_index, column_index].set_ylim(
                        shared_minimum,
                        shared_maximum,
                    )
                    axes[row_index, column_index].set_yticks(
                        shared_ticks
                    )

            axes[row_index, 0].set_ylabel(
                metric_label,
                fontsize=9,
                labelpad=7,
            )
        for row_index in range(4):
            axes[row_index, 0].yaxis.set_label_coords(-0.30, 0.5)
        figure.canvas.draw()

        for dataset_name, dataset_label, first_row in (
            ("fashionmnist", "FashionMNIST", 0),
            ("cifar10", "CIFAR-10", 2),
        ):
            top_position = axes[first_row, 0].get_position()
            bottom_position = axes[first_row + 1, 0].get_position()

            dataset_center = (
                top_position.y1 + bottom_position.y0
            ) / 2

            figure.text(
                0.035,
                dataset_center,
                dataset_label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
            )

        for axis in axes[:-1, :].flat:
            axis.tick_params(axis="x", labelbottom=False)

        figure.text(
            0.575,
            0.035,
            "Epoch",
            ha="center",
            va="center",
            fontsize=10,
        )

        figure.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    color=train_color,
                    linewidth=1.5,
                    label="Train",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=test_color,
                    linewidth=1.5,
                    label="Test",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        output_path = figure_path.with_name(
            f"{figure_path.stem}_{model_name}"
        )

        figure.savefig(
            output_path.with_suffix(".pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        figure.savefig(
            output_path.with_suffix(".png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

        print(f"Saved {output_path.with_suffix('.pdf')}")
        print(f"Saved {output_path.with_suffix('.png')}")

    print(f"Saved configuration mapping: {configuration_path}")


def main():
    """Parse paths and generate the training-results figure."""
    argument_parser = argparse.ArgumentParser(
        description="Plot MLP and CNN results grouped by configuration."
    )

    argument_parser.add_argument(
        "--log-directory",
        type=Path,
        # required=True,
        default=LOG_DIR,
    )

    argument_parser.add_argument(
        "--figure-path",
        type=Path,
        # required=True,
        default=FIG_PATH,
    )

    arguments = argument_parser.parse_args()

    plot_training_results(
        arguments.log_directory,
        arguments.figure_path,
    )


if __name__ == "__main__":
    main()
