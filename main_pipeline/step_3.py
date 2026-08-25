import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# Directory containing the evaluation files from step_2_evaluation.py
EVALUATION_INPUT_DIR = BASE_DIR / "step_2_output"

# Directory for visualization output files
VISUALIZATION_OUTPUT_DIR = BASE_DIR / "step_3_output"

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
    "exact_matches": re.compile(r"Exact matches:\s*(\d+)"),
    "matches_within_1": re.compile(
        r"Matches within \+/-1 paragraph:\s*(\d+)"
    ),
    "matches_within_2": re.compile(
        r"Matches within \+/-2 paragraphs:\s*(\d+)"
    ),
    "matches_within_3": re.compile(
        r"Matches within \+/-3 paragraphs:\s*(\d+)"
    ),
    "matches_within_5": re.compile(
        r"Matches within \+/-5 paragraphs:\s*(\d+)"
    ),
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


def calculate_match_share(matches, ground_truth_boundaries):
    """Calculate the share of ground truth boundaries covered by matches."""
    if ground_truth_boundaries == 0:
        return 0

    return matches / ground_truth_boundaries


def parse_evaluation_file(file_path):
    """Parse one evaluation file and return its metrics."""
    text = file_path.read_text(encoding="utf-8")
    w_value, k_value = get_parameter_values(file_path)

    true_positives = parse_metric(text, "true_positives")
    false_positives = parse_metric(text, "false_positives")
    false_negatives = parse_metric(text, "false_negatives")
    ground_truth_boundaries = parse_metric(text, "ground_truth_boundaries")

    exact_matches = parse_metric(text, "exact_matches")
    matches_within_1 = parse_metric(text, "matches_within_1")
    matches_within_2 = parse_metric(text, "matches_within_2")
    matches_within_3 = parse_metric(text, "matches_within_3")
    matches_within_5 = parse_metric(text, "matches_within_5")

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
        "ground_truth_boundaries": ground_truth_boundaries,
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
        "exact_matches": exact_matches,
        "matches_within_1": matches_within_1,
        "matches_within_2": matches_within_2,
        "matches_within_3": matches_within_3,
        "matches_within_5": matches_within_5,
        "exact_match_share": calculate_match_share(
            exact_matches,
            ground_truth_boundaries,
        ),
        "within_1_share": calculate_match_share(
            matches_within_1,
            ground_truth_boundaries,
        ),
        "within_2_share": calculate_match_share(
            matches_within_2,
            ground_truth_boundaries,
        ),
        "within_3_share": calculate_match_share(
            matches_within_3,
            ground_truth_boundaries,
        ),
        "within_5_share": calculate_match_share(
            matches_within_5,
            ground_truth_boundaries,
        ),
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


def create_distance_plot(results, output_path):
    """Create a plot for cumulative distance-based matches."""
    labels = [result["parameter"] for result in results]

    exact_values = [result["exact_matches"] for result in results]
    within_1_values = [result["matches_within_1"] for result in results]
    within_2_values = [result["matches_within_2"] for result in results]
    within_3_values = [result["matches_within_3"] for result in results]
    within_5_values = [result["matches_within_5"] for result in results]

    plt.figure(figsize=(14, 7))
    plt.plot(labels, exact_values, marker="o", label="Exact")
    plt.plot(labels, within_1_values, marker="o", label="+/-1")
    plt.plot(labels, within_2_values, marker="o", label="+/-2")
    plt.plot(labels, within_3_values, marker="o", label="+/-3")
    plt.plot(labels, within_5_values, marker="o", label="+/-5")

    for index, value in enumerate(within_5_values):
        plt.annotate(
            str(value),
            (index, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    plt.title("Distance-based boundary matches by parameter setting")
    plt.xlabel("Parameter setting")
    plt.ylabel("Number of matched ground truth boundaries")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_distance_curve_plot(results, output_path):
    """
    Create a readable curve for distance thresholds.

    The available data is cumulative, so this plot shows how many ground
    truth boundaries are covered at increasing distance thresholds.
    """
    plt.figure(figsize=(12, 7))
    thresholds = [0, 1, 2, 3, 5]

    for result in results:
        values = [
            result["exact_matches"],
            result["matches_within_1"],
            result["matches_within_2"],
            result["matches_within_3"],
            result["matches_within_5"],
        ]

        plt.plot(
            thresholds,
            values,
            marker="o",
            label=result["parameter"],
        )

        for threshold, value in zip(thresholds, values):
            plt.annotate(
                str(value),
                (threshold, value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )

    plt.title("Cumulative boundary matches by distance threshold")
    plt.xlabel("Maximum distance in paragraphs")
    plt.ylabel("Matched ground truth boundaries")
    plt.xticks(thresholds)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=8, ncol=2)
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
        "exact_matches",
        "matches_within_1",
        "matches_within_2",
        "matches_within_3",
        "matches_within_5",
        "exact_match_share",
        "within_1_share",
        "within_2_share",
        "within_3_share",
        "within_5_share",
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
            row["exact_match_share"] = f"{row['exact_match_share']:.4f}"
            row["within_1_share"] = f"{row['within_1_share']:.4f}"
            row["within_2_share"] = f"{row['within_2_share']:.4f}"
            row["within_3_share"] = f"{row['within_3_share']:.4f}"
            row["within_5_share"] = f"{row['within_5_share']:.4f}"
            writer.writerow(row)


def print_summary(
    results,
    metric_plot_file,
    distance_plot_file,
    distance_curve_file,
    table_file,
):
    """Print a short summary for the generated files."""
    print("Visualization completed.")
    print(f"Processed evaluation files: {len(results)}")
    print(f"Metric plot saved to: {metric_plot_file}")
    print(f"Distance plot saved to: {distance_plot_file}")
    print(f"Distance curve saved to: {distance_curve_file}")
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
    metric_plot_file = output_dir / "precision_recall_f1_plot.png"
    distance_plot_file = output_dir / "distance_matches_plot.png"
    distance_curve_file = output_dir / "distance_threshold_curve.png"
    table_file = output_dir / "evaluation_results_table.csv"

    create_metric_plot(results, metric_plot_file)
    create_distance_plot(results, distance_plot_file)
    create_distance_curve_plot(results, distance_curve_file)
    write_results_table(results, table_file)

    print_summary(
        results,
        metric_plot_file,
        distance_plot_file,
        distance_curve_file,
        table_file,
    )


if __name__ == "__main__":
    main()
