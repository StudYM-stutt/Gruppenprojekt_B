import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


# Directory containing the evaluation files from step_2_evaluation.py
EVALUATION_INPUT_DIR = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Evaluation_step2"
)

# Directory for visualization output files
VISUALIZATION_OUTPUT_DIR = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Visualization_step3"
)

EVALUATION_FILE_PATTERN = "evaluation_output_step1_w*_k*.txt"
PARAMETER_PATTERN = re.compile(
    r"evaluation_output_step1_w(\d+)_k(\d+)\.txt$"
)

METRIC_PATTERNS = {
    "ground_truth_boundaries": re.compile(
        r"Read ground truth boundaries:\s*(\d+)"
    ),
    "predicted_boundaries": re.compile(r"Predicted boundaries:\s*(\d+)"),
    "true_positives": re.compile(r"TP:\s*(\d+)"),
    "false_positives": re.compile(r"FP:\s*(\d+)"),
    "false_negatives": re.compile(r"FN:\s*(\d+)"),
    "precision": re.compile(r"Precision:\s*([0-9.]+)"),
    "recall": re.compile(r"Recall:\s*([0-9.]+)"),
    "f1_score": re.compile(r"F1 score:\s*([0-9.]+)"),
}


def get_evaluation_files():
    """Return all matching evaluation files."""
    input_dir = Path(EVALUATION_INPUT_DIR)
    files = sorted(input_dir.glob(EVALUATION_FILE_PATTERN))

    return [
        file_path
        for file_path in files
        if PARAMETER_PATTERN.match(file_path.name)
    ]


def get_parameter_values(file_path):
    """Extract w and k values from an evaluation file name."""
    match = PARAMETER_PATTERN.match(file_path.name)

    if not match:
        raise ValueError(f"Invalid evaluation file name: {file_path.name}")

    return int(match.group(1)), int(match.group(2))


def parse_metric(text, metric_name):
    """Parse one metric from an evaluation file."""
    pattern = METRIC_PATTERNS[metric_name]
    match = pattern.search(text)

    if not match:
        raise ValueError(f"Metric not found: {metric_name}")

    value = match.group(1)

    if "." in value:
        return float(value)

    return int(value)


def calculate_jaccard_index(
    true_positives,
    false_positives,
    false_negatives,
):
    """
    Calculate the Jaccard index.

    Jaccard index = TP / (TP + FP + FN)
    """
    denominator = true_positives + false_positives + false_negatives

    if denominator == 0:
        return 0

    return true_positives / denominator


def parse_evaluation_file(file_path):
    """Parse one evaluation file and return its metrics."""
    text = file_path.read_text(encoding="utf-8")
    w_value, k_value = get_parameter_values(file_path)

    true_positives = parse_metric(text, "true_positives")
    false_positives = parse_metric(text, "false_positives")
    false_negatives = parse_metric(text, "false_negatives")

    jaccard_index = calculate_jaccard_index(
        true_positives,
        false_positives,
        false_negatives,
    )

    return {
        "source_file": file_path.name,
        "parameter": f"w{w_value}_k{k_value}",
        "w": w_value,
        "k": k_value,
        "ground_truth_boundaries": parse_metric(
            text,
            "ground_truth_boundaries",
        ),
        "predicted_boundaries": parse_metric(
            text,
            "predicted_boundaries",
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": parse_metric(text, "precision"),
        "recall": parse_metric(text, "recall"),
        "f1_score": parse_metric(text, "f1_score"),
        "jaccard_index": jaccard_index,
    }


def load_evaluation_results():
    """Load all evaluation results."""
    evaluation_files = get_evaluation_files()

    if not evaluation_files:
        raise FileNotFoundError("No matching evaluation files were found.")

    results = [
        parse_evaluation_file(file_path)
        for file_path in evaluation_files
    ]

    return sorted(results, key=lambda result: (result["w"], result["k"]))


def create_metric_plot(results, output_path):
    """Create a plot for precision, recall, and F1 score."""
    labels = [result["parameter"] for result in results]
    precision_values = [result["precision"] for result in results]
    recall_values = [result["recall"] for result in results]
    f1_values = [result["f1_score"] for result in results]

    plt.figure(figsize=(12, 6))
    plt.plot(labels, precision_values, marker="o", label="Precision")
    plt.plot(labels, recall_values, marker="o", label="Recall")
    plt.plot(labels, f1_values, marker="o", label="F1 score")

    plt.title("TextTiling evaluation by parameter setting")
    plt.xlabel("Parameter setting")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_results_table(results, output_path):
    """Write the evaluation results as a CSV table."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "parameter",
        "w",
        "k",
        "ground_truth_boundaries",
        "predicted_boundaries",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "jaccard_index",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for result in results:
            row = result.copy()
            row["precision"] = f"{row['precision']:.4f}"
            row["recall"] = f"{row['recall']:.4f}"
            row["f1_score"] = f"{row['f1_score']:.4f}"
            row["jaccard_index"] = f"{row['jaccard_index']:.4f}"
            writer.writerow(row)


def print_summary(results, plot_file, table_file):
    """Print a short summary for the generated files."""
    print("Visualization completed.")
    print(f"Processed evaluation files: {len(results)}")
    print(f"Metric plot saved to: {plot_file}")
    print(f"Result table saved to: {table_file}")
    print()

    for result in results:
        print(
            "Processed "
            f"{result['source_file']} "
            f"({result['parameter']})."
        )


def main():
    """Create visualizations for all step 2 evaluation files."""
    results = load_evaluation_results()

    output_dir = Path(VISUALIZATION_OUTPUT_DIR)
    plot_file = output_dir / "precision_recall_f1_plot.png"
    table_file = output_dir / "evaluation_results_table.csv"

    create_metric_plot(results, plot_file)
    write_results_table(results, table_file)
    print_summary(results, plot_file, table_file)


if __name__ == "__main__":
    main()
