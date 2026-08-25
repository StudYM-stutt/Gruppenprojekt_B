import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# Directory containing the evaluation files from step_2_llm.py.
# The search is recursive because step_2 writes files into nested folders like:
# step_2_output/variant_01/llm_ollama_qwen3_8b_batched/Prompt_01/evaluation_output_step2_Prompt_01_bs10_cs3.txt
EVALUATION_INPUT_DIR = BASE_DIR / "step_2_output"

# Directory for Step 3 visualization output files.
VISUALIZATION_OUTPUT_DIR = BASE_DIR / "step_3_output"

EVALUATION_FILE_PATTERN = "evaluation_output_step2_*.txt"
PARAMETER_PATTERN = re.compile(
    r"evaluation_output_step2_(?P<prompt>.+)_(?P<parameter>(?:bs\d+_cs\d+)|(?:bw\d+-\d+_cw\d+))\.txt$"
)

# Extract provider and model from run directories such as:
# llm_dryrun_gpt-4o-mini_batched
# llm_ollama_qwen3_8b_batched
# This stays extensible because everything between provider and _batched is treated as model name.
RUN_DIR_PATTERN = re.compile(r"^llm_(?P<provider>[^_]+)_(?P<model>.+)_batched$")

METRIC_PATTERNS = {
    "ground_truth_boundaries": re.compile(r"Read ground truth boundaries:\s*(\d+)"),
    "predicted_boundaries": re.compile(r"Predicted boundaries:\s*(\d+)"),
    "true_positives": re.compile(r"TP:\s*(\d+)"),
    "false_positives": re.compile(r"FP:\s*(\d+)"),
    "false_negatives": re.compile(r"FN:\s*(\d+)"),
    "precision": re.compile(r"Precision:\s*([0-9.]+)"),
    "recall": re.compile(r"Recall:\s*([0-9.]+)"),
    "f1_score": re.compile(r"F1 score:\s*([0-9.]+)"),
    "exact_matches": re.compile(r"Exact matches:\s*(\d+)"),
    "matches_within_1": re.compile(r"Matches within \+/-1 paragraph:\s*(\d+)"),
    "matches_within_2": re.compile(r"Matches within \+/-2 paragraphs:\s*(\d+)"),
    "matches_within_3": re.compile(r"Matches within \+/-3 paragraphs:\s*(\d+)"),
    "matches_within_5": re.compile(r"Matches within \+/-5 paragraphs:\s*(\d+)"),
}


def slugify(value):
    """Create a safe file-name component."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def prompt_sort_key(prompt_name):
    """Sort prompts robustly, even when names mix numeric and descriptive forms."""
    match = re.search(r"Prompt[_-]?(\d+)", prompt_name, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), prompt_name.lower())
    return (1, 0, prompt_name.lower())


def get_evaluation_files():
    """Return all matching Step 2 evaluation files recursively."""
    files = sorted(EVALUATION_INPUT_DIR.rglob(EVALUATION_FILE_PATTERN))

    return [
        file_path
        for file_path in files
        if PARAMETER_PATTERN.match(file_path.name)
    ]


def get_parameter_values(file_path):
    """Extract prompt and batching parameters from a Step 2 evaluation filename."""
    match = PARAMETER_PATTERN.match(file_path.name)

    if not match:
        raise ValueError(f"Invalid evaluation file name: {file_path.name}")

    prompt_name = match.group("prompt")
    parameter = match.group("parameter")

    if parameter.startswith("bs"):
        return {
            "prompt_name": prompt_name,
            "parameter": parameter,
            "batch_mode": "paragraphs",
            "batch_size": int(re.search(r"bs(\d+)", parameter).group(1)),
            "context_size": int(re.search(r"cs(\d+)", parameter).group(1)),
            "min_batch_words": "",
            "max_batch_words": "",
            "min_context_words": "",
        }

    word_match = re.fullmatch(r"bw(\d+)-(\d+)_cw(\d+)", parameter)
    if not word_match:
        raise ValueError(f"Unsupported parameter suffix: {parameter}")

    return {
        "prompt_name": prompt_name,
        "parameter": parameter,
        "batch_mode": "words",
        "batch_size": "",
        "context_size": "",
        "min_batch_words": int(word_match.group(1)),
        "max_batch_words": int(word_match.group(2)),
        "min_context_words": int(word_match.group(3)),
    }


def get_run_values(file_path):
    """Extract provider/model from the run directory if possible."""
    for parent in file_path.parents:
        match = RUN_DIR_PATTERN.match(parent.name)
        if match:
            return match.group("provider"), match.group("model")

    return "unknown", "unknown"


def get_variant(file_path):
    """Extract the synthetic variant folder name, e.g. variant_01."""
    for parent in file_path.parents:
        if parent.name.startswith("variant_"):
            return parent.name

    return "unknown_variant"


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


def calculate_jaccard_index(true_positives, false_positives, false_negatives):
    """Calculate the Jaccard index: TP / (TP + FP + FN)."""
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
    """Parse one Step 2 evaluation file and return its metrics."""
    text = file_path.read_text(encoding="utf-8")
    parameter_values = get_parameter_values(file_path)
    prompt_name = parameter_values["prompt_name"]
    parameter_suffix = parameter_values["parameter"]
    provider, model = get_run_values(file_path)
    variant = get_variant(file_path)

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

    parameter = f"{model}_{prompt_name}_{parameter_suffix}"
    group_name = f"{variant}_{model}_{parameter_suffix}"

    return {
        "source_file": str(file_path.relative_to(EVALUATION_INPUT_DIR)),
        "variant": variant,
        "prompt_name": prompt_name,
        "provider": provider,
        "model": model,
        "parameter": parameter,
        "group_name": group_name,
        "batch_mode": parameter_values["batch_mode"],
        "batch_size": parameter_values["batch_size"],
        "context_size": parameter_values["context_size"],
        "min_batch_words": parameter_values["min_batch_words"],
        "max_batch_words": parameter_values["max_batch_words"],
        "min_context_words": parameter_values["min_context_words"],
        "parameter_suffix": parameter_suffix,
        "ground_truth_boundaries": ground_truth_boundaries,
        "predicted_boundaries": parse_metric(text, "predicted_boundaries"),
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
        "exact_match_share": calculate_match_share(exact_matches, ground_truth_boundaries),
        "within_1_share": calculate_match_share(matches_within_1, ground_truth_boundaries),
        "within_2_share": calculate_match_share(matches_within_2, ground_truth_boundaries),
        "within_3_share": calculate_match_share(matches_within_3, ground_truth_boundaries),
        "within_5_share": calculate_match_share(matches_within_5, ground_truth_boundaries),
    }


def load_evaluation_results():
    """Load all Step 2 evaluation results."""
    evaluation_files = get_evaluation_files()

    if not evaluation_files:
        raise FileNotFoundError(
            f"No matching Step 2 evaluation files were found in: {EVALUATION_INPUT_DIR}\n"
            f"Expected pattern: {EVALUATION_FILE_PATTERN}"
        )

    results = [parse_evaluation_file(file_path) for file_path in evaluation_files]

    return sorted(
        results,
        key=lambda result: (
            result["variant"],
            result["model"],
            result["parameter_suffix"],
            prompt_sort_key(result["prompt_name"]),
        ),
    )


def group_results(results):
    """Group results by identical variant, model, batch size and context size."""
    grouped_results = defaultdict(list)

    for result in results:
        group_key = (
            result["variant"],
            result["model"],
            result["parameter_suffix"],
        )
        grouped_results[group_key].append(result)

    return grouped_results


def make_group_output_dir(output_dir, group_key):
    """Create one output subfolder per variant + model + parameter setting."""
    variant, model, parameter_suffix = group_key
    group_dir = output_dir / variant / f"{slugify(model)}_{slugify(parameter_suffix)}"
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def create_metric_plot(results, output_path, title_prefix):
    """Create a grouped plot for precision, recall and F1 score."""
    labels = [result["parameter"] for result in results]
    precision_values = [result["precision"] for result in results]
    recall_values = [result["recall"] for result in results]
    f1_values = [result["f1_score"] for result in results]

    plt.figure(figsize=(14, 7))
    plt.plot(labels, precision_values, marker="o", label="Precision")
    plt.plot(labels, recall_values, marker="o", label="Recall")
    plt.plot(labels, f1_values, marker="o", label="F1 score")

    plt.title(f"{title_prefix}: Precision, Recall and F1")
    plt.xlabel("Prompt")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_distance_plot(results, output_path, title_prefix):
    """Create a grouped plot for cumulative distance-based matches."""
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

    plt.title(f"{title_prefix}: Distance-based boundary matches")
    plt.xlabel("Prompt")
    plt.ylabel("Number of matched ground truth boundaries")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_distance_curve_plot(results, output_path, title_prefix):
    """Create one grouped curve plot for distance thresholds."""
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

        plt.plot(thresholds, values, marker="o", label=result["parameter"])

        for threshold, value in zip(thresholds, values):
            plt.annotate(
                str(value),
                (threshold, value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )

    plt.title(f"{title_prefix}: Cumulative boundary matches by distance threshold")
    plt.xlabel("Maximum distance in paragraphs")
    plt.ylabel("Matched ground truth boundaries")
    plt.xticks(thresholds)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_results_table(results, output_path):
    """Write the evaluation results as a CSV table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "variant",
        "prompt_name",
        "provider",
        "model",
        "parameter",
        "group_name",
        "batch_mode",
        "batch_size",
        "context_size",
        "min_batch_words",
        "max_batch_words",
        "min_context_words",
        "parameter_suffix",
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

    with output_path.open("w", encoding="utf-8", newline="") as file:
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


def write_grouped_outputs(grouped_results, output_dir):
    """Write plots and tables for every model + bs/cs group."""
    created_files = []

    for group_key, group in sorted(grouped_results.items()):
        variant, model, parameter_suffix = group_key
        group = sorted(group, key=lambda result: prompt_sort_key(result["prompt_name"]))
        group_dir = make_group_output_dir(output_dir, group_key)
        title_prefix = f"{variant} {model} {parameter_suffix}"
        file_prefix = f"{slugify(variant)}_{slugify(model)}_{slugify(parameter_suffix)}"

        metric_plot_file = group_dir / f"{file_prefix}_precision_recall_f1_plot.png"
        distance_plot_file = group_dir / f"{file_prefix}_distance_matches_plot.png"
        distance_curve_file = group_dir / f"{file_prefix}_distance_threshold_curve.png"
        table_file = group_dir / f"{file_prefix}_evaluation_results_table.csv"

        create_metric_plot(group, metric_plot_file, title_prefix)
        create_distance_plot(group, distance_plot_file, title_prefix)
        create_distance_curve_plot(group, distance_curve_file, title_prefix)
        write_results_table(group, table_file)

        created_files.extend([
            metric_plot_file,
            distance_plot_file,
            distance_curve_file,
            table_file,
        ])

    return created_files


def print_summary(results, grouped_results, created_files, full_table_file):
    """Print a short summary for the generated files."""
    print("Step 3 visualization completed.")
    print(f"Processed evaluation files: {len(results)}")
    print(f"Created model/parameter groups: {len(grouped_results)}")
    print(f"Complete result table saved to: {full_table_file}")
    print()

    for group_key, group in sorted(grouped_results.items()):
        variant, model, parameter_suffix = group_key
        print(
            f"Group {variant}_{model}_{parameter_suffix}: "
            f"{len(group)} prompt file(s)"
        )

    print("\nCreated output files:")
    for file_path in created_files:
        print(file_path)


def main():
    """Create grouped visualizations for all Step 2 evaluation files."""
    results = load_evaluation_results()
    grouped_results = group_results(results)

    output_dir = Path(VISUALIZATION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_table_file = output_dir / "step2_all_evaluation_results_table.csv"
    write_results_table(results, full_table_file)

    created_files = write_grouped_outputs(grouped_results, output_dir)
    created_files.insert(0, full_table_file)

    print_summary(results, grouped_results, created_files, full_table_file)


if __name__ == "__main__":
    main()
