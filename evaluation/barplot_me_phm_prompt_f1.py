from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

TARGET_FILES = [
    "ME_Phase_1.csv",
    "ME_Phase_2.csv",
    "PHM_Phase_1.csv",
    "PHM_Phase_2.csv",
]

PROMPT_FAMILY_COLORS = {
    "baseline": "purple",
    "plot_frame": "brown",
    "protagonist_frame": "red",
    "situation_model": "blue",
    "event_frame": "green",
    "goal_frame": "orange",
}

FALLBACK_COLORS = [
    "cyan",
    "magenta",
    "olive",
    "pink",
    "gray",
    "yellow",
]


# ==========================================================
# Helper functions
# ==========================================================

def read_csv_robustly(path):
    """
    Read comma-, semicolon-, or tab-separated CSV files automatically.
    """
    return pd.read_csv(path, sep=None, engine="python")


def prompt_sort_key(prompt_name):
    """
    Sort prompts by their leading prompt number.
    """
    match = re.search(r"Prompt_(\d+)([A-Za-z]?)", str(prompt_name))

    if not match:
        return (9999, "", str(prompt_name))

    number = int(match.group(1))
    suffix = match.group(2).lower()

    return (number, suffix, str(prompt_name))


def get_prompt_family(prompt_name):
    """
    Map prompt names to stable semantic prompt families.
    """
    name = str(prompt_name).lower()

    if "baseline" in name:
        return "baseline"
    if "plot_frame" in name:
        return "plot_frame"
    if "protagonist_frame" in name:
        return "protagonist_frame"
    if "situation_model" in name:
        return "situation_model"
    if "event_frame" in name:
        return "event_frame"
    if "goal_frame" in name:
        return "goal_frame"

    return None


def build_prompt_color_map(prompt_names):
    """
    Assign stable colors to known prompt families and fallback colors to
    previously unseen prompts.
    """
    colors = {}
    fallback_index = 0

    for prompt in prompt_names:
        family = get_prompt_family(prompt)

        if family in PROMPT_FAMILY_COLORS:
            colors[prompt] = PROMPT_FAMILY_COLORS[family]
        else:
            colors[prompt] = FALLBACK_COLORS[
                fallback_index % len(FALLBACK_COLORS)
            ]
            fallback_index += 1

    return colors


def format_prompt_label(prompt_name):
    """
    Convert a prompt identifier into a compact readable axis label.
    """
    text = str(prompt_name)
    match = re.match(r"Prompt_(\d+[A-Za-z]?)_(.+)", text)

    if not match:
        return text

    number = match.group(1)
    description = match.group(2).replace("_", " ")
    description = " ".join(
        word if word.isupper() else word.capitalize()
        for word in description.split()
    )

    return f"{number}\n{description}"


def get_dataset_label(input_file):
    """
    Use the CSV file name without its extension as the dataset label.
    """
    return input_file.stem


# ==========================================================
# Plotting
# ==========================================================

def create_plot(input_file):
    """
    Create one vertical bar chart showing the F1 score of each prompt.
    """
    df = read_csv_robustly(input_file)

    required_columns = {"prompt_name", "f1_score"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_file.name}: Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["prompt_name"] = df["prompt_name"].astype(str).str.strip()
    df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")
    df = df.dropna(subset=["prompt_name", "f1_score"]).copy()

    if df.empty:
        raise ValueError(
            f"{input_file.name}: No valid prompt/F1-score rows found."
        )

    # If a prompt occurs more than once, show its mean F1 score.
    plot_df = (
        df.groupby("prompt_name", as_index=False)["f1_score"]
        .mean()
    )

    prompt_names = sorted(
        plot_df["prompt_name"].tolist(),
        key=prompt_sort_key,
    )

    plot_df["prompt_name"] = pd.Categorical(
        plot_df["prompt_name"],
        categories=prompt_names,
        ordered=True,
    )
    plot_df = plot_df.sort_values("prompt_name")

    prompt_colors = build_prompt_color_map(prompt_names)
    colors = [
        prompt_colors[str(prompt)]
        for prompt in plot_df["prompt_name"]
    ]

    x_positions = list(range(len(plot_df)))

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        x_positions,
        plot_df["f1_score"],
        width=0.68,
        color=colors,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
    )

    # Use the same fixed scale for all datasets.
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([i * 0.05 for i in range(11)])

    # Add horizontal guide lines at every 0.05 F1-score step.
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            format_prompt_label(str(prompt))
            for prompt in plot_df["prompt_name"]
        ]
    )

    ax.set_xlabel("Promptvariation")
    ax.set_ylabel("F1-Score")
    ax.set_title(
        "F1-Score der Promptvarianten "
        f"({get_dataset_label(input_file)})"
    )

    # Show the exact F1 score above each bar.
    for bar, value in zip(bars, plot_df["f1_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = input_file.with_name(
        f"{input_file.stem}_barplot.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{input_file.name}: Bar chart saved to: {output_file}"
    )


# ==========================================================
# Input handling
# ==========================================================

def get_input_files():
    """
    Return only the intended ME and PHM CSV files.

    Command-line arguments take priority. Without arguments, the script
    searches for the configured target files in the script directory.
    """
    if len(sys.argv) > 1:
        files = []

        for argument in sys.argv[1:]:
            path = Path(argument)

            if not path.is_absolute():
                path = BASE_DIR / path

            files.append(path)

        return files

    return [
        BASE_DIR / filename
        for filename in TARGET_FILES
        if (BASE_DIR / filename).exists()
    ]


def main():
    """
    Create one bar chart for each available target CSV file.
    """
    input_files = get_input_files()

    if not input_files:
        raise FileNotFoundError(
            "No target CSV files found. Expected: "
            + ", ".join(TARGET_FILES)
        )

    errors = []

    for input_file in input_files:
        try:
            create_plot(input_file)
        except Exception as exc:
            errors.append((input_file.name, str(exc)))
            print(f"ERROR in {input_file.name}: {exc}")

    if errors:
        print("\nSome files could not be processed:")

        for filename, error in errors:
            print(f"  - {filename}: {error}")


if __name__ == "__main__":
    main()
