from pathlib import Path


# Output from step_1_texttiling.py
STEP1_OUTPUT_FILE = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Output\output_step1.txt"
)

# Ground truth file
GROUND_TRUTH_FILE = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Grenzen_unendliche_Geschichte"
    r"\grenzen_ground_truth_unendliche_geschichte.txt"
)

# Evaluation output file
EVALUATION_OUTPUT_FILE = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Evaluation_step2"
    r"\output_evaluation_step2.txt"
)


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


def save_results(
    path,
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
):
    """Save evaluation results to a text file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("=== Evaluation TextTiling ===\n\n")

        file.write(f"Step 1 output: {STEP1_OUTPUT_FILE}\n")
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

        file.write("True positive boundaries:\n")
        for boundary in sorted(true_positive_boundaries):
            file.write(f"{boundary}\n")


def print_results(
    total_ground_truth_boundaries,
    predicted_boundaries,
    ground_truth_boundaries,
    true_positives,
    false_positives,
    false_negatives,
    precision,
    recall,
    f1_score,
):
    """Print evaluation results to the console."""
    print("=== Evaluation TextTiling ===")
    print(f"Step 1 output: {STEP1_OUTPUT_FILE}")
    print(f"Ground truth: {GROUND_TRUTH_FILE}")
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

    print(f"Results saved to: {EVALUATION_OUTPUT_FILE}")


def main():
    """Run the evaluation."""
    total_ground_truth_boundaries, ground_truth_boundaries = (
        load_ground_truth(GROUND_TRUTH_FILE)
    )

    predicted_boundaries = extract_predicted_boundaries(STEP1_OUTPUT_FILE)

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
        EVALUATION_OUTPUT_FILE,
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
    )

    print_results(
        total_ground_truth_boundaries,
        predicted_boundaries,
        ground_truth_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    )


if __name__ == "__main__":
    main()