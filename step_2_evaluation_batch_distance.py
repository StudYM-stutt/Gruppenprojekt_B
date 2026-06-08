import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Directory containing the output files from step_1_texttiling.py

STEP1_OUTPUT_DIR = BASE_DIR / "step_1_output"

# Ground truth file
GROUND_TRUTH_FILE = (
    BASE_DIR
    / "Grenzen"
    / "g_grenzen.txt"
)

# Directory for evaluation output files
EVALUATION_OUTPUT_DIR = BASE_DIR / "step_2_output"

STEP1_OUTPUT_PATTERN = "output_step1_w*_k*.txt"
PARAMETER_PATTERN = re.compile(r"output_step1_w(\d+)_k(\d+)\.txt$")


def load_ground_truth(path):
    """
    Read the ground truth file.

    Expected format:
    First line: total number of boundaries.
    Following lines: paragraph numbers where a boundary begins.
    """
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    total_boundaries = int(lines[0])
    boundaries = {int(line) for line in lines[1:]}

    return total_boundaries, boundaries


def extract_predicted_boundaries(path):
    """
    Read predicted segment boundaries from the step 1 output.

    The first paragraph after a segment heading is treated as a boundary.
    Segment 1 starts at paragraph 0 and is not counted as a boundary.
    """
    boundaries = set()

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    waiting_for_first_paragraph = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("--- Segment"):
            waiting_for_first_paragraph = True
            continue

        if waiting_for_first_paragraph and stripped.startswith("["):
            try:
                paragraph_id = int(stripped.split("]")[0][1:])
                boundaries.add(paragraph_id)
                waiting_for_first_paragraph = False
            except ValueError:
                pass

    boundaries.discard(0)
    return boundaries


def evaluate(predicted_boundaries, ground_truth_boundaries):
    """Calculate precision, recall, and F1 score."""
    true_positive_boundaries = predicted_boundaries & ground_truth_boundaries

    true_positives = len(true_positive_boundaries)
    false_positives = len(predicted_boundaries - ground_truth_boundaries)
    false_negatives = len(ground_truth_boundaries - predicted_boundaries)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else 0
    )

    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return (
        true_positive_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    )


def calculate_boundary_distances(
    predicted_boundaries,
    ground_truth_boundaries,
):
    """Calculate distances to the nearest predicted boundary."""
    if not predicted_boundaries:
        return []

    distances = []

    for ground_truth_boundary in sorted(ground_truth_boundaries):
        nearest_prediction = min(
            predicted_boundaries,
            key=lambda boundary: abs(boundary - ground_truth_boundary),
        )
        distance = abs(nearest_prediction - ground_truth_boundary)
        distances.append(distance)

    return distances


def calculate_distance_matches(distances):
    """Calculate matches within selected paragraph distance thresholds."""
    return {
        "exact_matches": sum(distance == 0 for distance in distances),
        "matches_within_1": sum(distance <= 1 for distance in distances),
        "matches_within_2": sum(distance <= 2 for distance in distances),
        "matches_within_3": sum(distance <= 3 for distance in distances),
        "matches_within_5": sum(distance <= 5 for distance in distances),
    }


def get_parameter_values(file_path):
    """Extract w and k values from a step 1 output file name."""
    match = PARAMETER_PATTERN.match(file_path.name)

    if not match:
        return None

    return match.group(1), match.group(2)


def create_evaluation_output_file(w_value, k_value):
    """Create the evaluation output path for one parameter setting."""
    file_name = f"evaluation_output_step1_w{w_value}_k{k_value}.txt"
    return Path(EVALUATION_OUTPUT_DIR) / file_name


def save_results(
    path,
    step1_output_file,
    total_ground_truth_boundaries,
    predicted_boundaries,
    ground_truth_boundaries,
    true_positive_boundaries,
    true_positives,
    false_positives,
    false_negatives,
    precision,
    recall,
    f1_score,
    distance_matches,
):
    """Save evaluation results to a text file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("=== Evaluation TextTiling ===\n\n")

        file.write(f"Step 1 output: {step1_output_file}\n")
        file.write(f"Ground truth: {GROUND_TRUTH_FILE}\n\n")

        file.write(
            "Ground truth boundaries according to file: "
            f"{total_ground_truth_boundaries}\n"
        )
        file.write(
            "Read ground truth boundaries: "
            f"{len(ground_truth_boundaries)}\n"
        )
        file.write(
            "Predicted boundaries: "
            f"{len(predicted_boundaries)}\n\n"
        )

        file.write(f"TP: {true_positives}\n")
        file.write(f"FP: {false_positives}\n")
        file.write(f"FN: {false_negatives}\n\n")

        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"F1 score:  {f1_score:.4f}\n\n")

        file.write("Distance analysis\n")
        file.write("-----------------\n")
        file.write(
            f"Exact matches: {distance_matches['exact_matches']}\n"
        )
        file.write(
            "Matches within +/-1 paragraph: "
            f"{distance_matches['matches_within_1']}\n"
        )
        file.write(
            "Matches within +/-2 paragraphs: "
            f"{distance_matches['matches_within_2']}\n"
        )
        file.write(
            "Matches within +/-3 paragraphs: "
            f"{distance_matches['matches_within_3']}\n"
        )
        file.write(
            "Matches within +/-5 paragraphs: "
            f"{distance_matches['matches_within_5']}\n\n"
        )

        file.write("True positive boundaries:\n")
        for boundary in sorted(true_positive_boundaries):
            file.write(f"{boundary}\n")


def print_results(
    step1_output_file,
    evaluation_output_file,
    total_ground_truth_boundaries,
    predicted_boundaries,
    ground_truth_boundaries,
    true_positives,
    false_positives,
    false_negatives,
    precision,
    recall,
    f1_score,
    distance_matches,
):
    """Print evaluation results to the console."""
    print("Evaluation completed.")
    print(f"Step 1 output: {step1_output_file}")
    print(f"Ground truth: {GROUND_TRUTH_FILE}")
    print(f"Evaluation output: {evaluation_output_file}")
    print()

    print(
        "Ground truth boundaries according to file: "
        f"{total_ground_truth_boundaries}"
    )
    print(
        "Read ground truth boundaries: "
        f"{len(ground_truth_boundaries)}"
    )
    print(f"Predicted boundaries: {len(predicted_boundaries)}")
    print()

    print(f"TP: {true_positives}")
    print(f"FP: {false_positives}")
    print(f"FN: {false_negatives}")
    print()

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1_score:.4f}")
    print()

    print("Distance analysis")
    print("-----------------")
    print(f"Exact matches: {distance_matches['exact_matches']}")
    print(
        "Matches within +/-1 paragraph: "
        f"{distance_matches['matches_within_1']}"
    )
    print(
        "Matches within +/-2 paragraphs: "
        f"{distance_matches['matches_within_2']}"
    )
    print(
        "Matches within +/-3 paragraphs: "
        f"{distance_matches['matches_within_3']}"
    )
    print(
        "Matches within +/-5 paragraphs: "
        f"{distance_matches['matches_within_5']}"
    )
    print()


def get_step1_output_files():
    """Get all step 1 output files with w and k values in the file name."""
    output_dir = Path(STEP1_OUTPUT_DIR)
    files = sorted(output_dir.glob(STEP1_OUTPUT_PATTERN))

    return [
        file_path
        for file_path in files
        if get_parameter_values(file_path) is not None
    ]


def evaluate_file(
    step1_output_file,
    total_ground_truth_boundaries,
    ground_truth_boundaries,
):
    """Evaluate one step 1 output file."""
    w_value, k_value = get_parameter_values(step1_output_file)
    evaluation_output_file = create_evaluation_output_file(w_value, k_value)

    predicted_boundaries = extract_predicted_boundaries(step1_output_file)
    distances = calculate_boundary_distances(
        predicted_boundaries,
        ground_truth_boundaries,
    )
    distance_matches = calculate_distance_matches(distances)

    (
        true_positive_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    ) = evaluate(predicted_boundaries, ground_truth_boundaries)

    save_results(
        evaluation_output_file,
        step1_output_file,
        total_ground_truth_boundaries,
        predicted_boundaries,
        ground_truth_boundaries,
        true_positive_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
        distance_matches,
    )

    print_results(
        step1_output_file,
        evaluation_output_file,
        total_ground_truth_boundaries,
        predicted_boundaries,
        ground_truth_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
        distance_matches,
    )


def main():
    """Run the evaluation for all matching step 1 output files."""
    total_ground_truth_boundaries, ground_truth_boundaries = (
        load_ground_truth(GROUND_TRUTH_FILE)
    )

    step1_output_files = get_step1_output_files()

    if not step1_output_files:
        print("No matching step 1 output files were found.")
        return

    for step1_output_file in step1_output_files:
        evaluate_file(
            step1_output_file,
            total_ground_truth_boundaries,
            ground_truth_boundaries,
        )


if __name__ == "__main__":
    main()
