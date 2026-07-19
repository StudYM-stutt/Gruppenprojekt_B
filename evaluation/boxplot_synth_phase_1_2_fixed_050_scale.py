from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# Usage:
# 1) Without arguments:
#       python boxplot_synth_phase_1_2.py
#    -> Automatically processes Synth_Phase_1.csv and Synth_Phase_2.csv.
#
# 2) With one or more files:
#       python boxplot_synth_phase_1_2.py Synth_Phase_1.csv
#       python boxplot_synth_phase_1_2.py Synth_Phase_1.csv Synth_Phase_2.csv
#
# A PNG file with the same base name is created for each CSV file.

GRANULARITY_ORDER = [
    "Grob",
    "Mittel grob",
    "Fein",
    "Sehr fein",
]

GRANULARITY_DESCRIPTIONS = {
    "Grob": "<20",
    "Mittel grob": "20–38",
    "Fein": "39–99",
    "Sehr fein": "≥100",
}

# The same semantic prompt families retain the same color across
# Phase 1 and Phase 2. Unknown prompts are assigned a color automatically.
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

def get_granularity(ground_truth_boundaries):
    """
    Assign granularity based on the actual number of ground-truth boundaries.

    Grob:         fewer than 20 boundaries
    Mittel grob:  20 to 38 boundaries
    Fein:         39 to 99 boundaries
    Sehr fein:    100 or more boundaries
    """
    boundaries = int(float(ground_truth_boundaries))

    if boundaries < 20:
        return "Grob"
    if boundaries <= 38:
        return "Mittel grob"
    if boundaries < 100:
        return "Fein"
    return "Sehr fein"


def read_csv_robustly(path):
    """
    Read comma-, semicolon- or tab-separated CSV files automatically.
    """
    return pd.read_csv(path, sep=None, engine="python")


def prompt_sort_key(prompt_name):
    """
    Sort prompts by their leading prompt number, e.g.
    Prompt_01, Prompt_06a, Prompt_11, Prompt_11a, ...
    """
    match = re.search(r"Prompt_(\d+)([A-Za-z]?)", str(prompt_name))

    if not match:
        return (9999, "", str(prompt_name))

    number = int(match.group(1))
    suffix = match.group(2).lower()
    return (number, suffix, str(prompt_name))


def get_prompt_family(prompt_name):
    """
    Map Phase-1 and Phase-2 prompt names to stable semantic families.
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
    Use stable colors for known prompt families and automatic fallback colors
    for any previously unseen prompt.
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
    Convert e.g.
      Prompt_11a_Protagonist_Frame -> 11a – Protagonist Frame
      Prompt_11_Protagonist_Frame  -> 11 – Protagonist Frame
      Prompt_01_Baseline           -> 01 – Baseline
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

    return f"{number} – {description}"


def determine_n(df, group_mask=None):
    """
    Determine the number of independent data units.

    Priority:
    1. unique variants, if the synthetic 'variant' column exists
    2. unique source files
    3. number of rows
    """
    subset = df if group_mask is None else df.loc[group_mask]

    if "variant" in subset.columns and subset["variant"].notna().any():
        return subset["variant"].nunique()

    if "source_file" in subset.columns and subset["source_file"].notna().any():
        return subset["source_file"].nunique()

    return len(subset)


def has_meaningful_granularity(df):
    """
    Granularity is used only when ground-truth boundary counts actually
    produce more than one granularity group.

    This makes synthetic multi-variant CSVs use the four granularity groups,
    while single-document PHM/ME CSVs are shown as one overall group.
    """
    if "ground_truth_boundaries" not in df.columns:
        return False

    boundaries = pd.to_numeric(
        df["ground_truth_boundaries"],
        errors="coerce",
    ).dropna()

    if boundaries.empty:
        return False

    levels = boundaries.apply(get_granularity).nunique()
    return levels > 1


def set_dynamic_y_limits(ax, values):
    """
    Set a fixed F1-score scale from 0.00 to 0.50 in steps of 0.05.
    """
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([i * 0.05 for i in range(11)])


# ==========================================================
# Plotting
# ==========================================================

def create_plot(input_file):
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
    df = df.dropna(subset=["f1_score"]).copy()

    if df.empty:
        raise ValueError(
            f"{input_file.name}: No valid numeric F1 scores found."
        )

    # Detect all prompts dynamically; no prompt name has to be hard-coded.
    prompt_names = sorted(
        df["prompt_name"].dropna().unique(),
        key=prompt_sort_key,
    )

    if not prompt_names:
        raise ValueError(
            f"{input_file.name}: No prompt variants found."
        )

    prompt_colors = build_prompt_color_map(prompt_names)

    use_granularity = has_meaningful_granularity(df)

    if use_granularity:
        df["granularity"] = pd.to_numeric(
            df["ground_truth_boundaries"],
            errors="coerce",
        ).apply(
            lambda value: get_granularity(value)
            if pd.notna(value)
            else None
        )

        available_groups = [
            group
            for group in GRANULARITY_ORDER
            if group in df["granularity"].dropna().unique()
        ]

        group_labels = []
        for group in available_groups:
            n = determine_n(
                df,
                df["granularity"] == group,
            )
            description = GRANULARITY_DESCRIPTIONS[group]
            group_labels.append(
                f"{group} ({description}), n={n}"
            )

        x_axis_label = "Granularitätsstufe"
        phase_match = re.search(r"Synth_Phase_([12])", input_file.stem)
        phase_label = (
            f"Synth_Phase_{phase_match.group(1)}"
            if phase_match
            else input_file.stem
        )
        title = (
            "F1-Score der Promptvarianten nach Granularitätsstufe "
            f"({phase_label})"
        )

    else:
        available_groups = ["Gesamt"]
        n = determine_n(df)
        group_labels = [f"Gesamt, n={n}"]
        x_axis_label = "Datensatz"
        title = "F1-Score der Promptvarianten"

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    number_of_prompts = len(prompt_names)

    # Keep the overall width of each group similar regardless of prompt count.
    total_group_width = 0.70
    box_width = min(0.16, total_group_width / max(number_of_prompts, 1))
    base_positions = list(range(len(available_groups)))

    for prompt_index, prompt_name in enumerate(prompt_names):
        boxplot_data = []
        positions = []

        offset = (
            prompt_index
            - (number_of_prompts - 1) / 2
        ) * box_width

        for group_index, group in enumerate(available_groups):
            if use_granularity:
                values = df[
                    (df["prompt_name"] == prompt_name)
                    & (df["granularity"] == group)
                ]["f1_score"].dropna()
            else:
                values = df[
                    df["prompt_name"] == prompt_name
                ]["f1_score"].dropna()

            # Matplotlib cannot draw a useful box for an entirely empty
            # prompt/group combination, so skip only that one position.
            if values.empty:
                continue

            boxplot_data.append(values.to_numpy())
            positions.append(group_index + offset)

        if not boxplot_data:
            continue

        box = ax.boxplot(
            boxplot_data,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        color = prompt_colors[prompt_name]

        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in box["medians"]:
            median.set_color("black")

    # Axis labels
    ax.set_xticks(base_positions)
    ax.set_xticklabels(group_labels)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("F1-Score")
    ax.set_title(title)

    set_dynamic_y_limits(ax, df["f1_score"])

    # Add dashed horizontal guide lines at the y-axis tick positions.
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    # First legend: colors identify prompt variants.
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=prompt_colors[prompt],
            alpha=0.6,
        )
        for prompt in prompt_names
    ]

    legend_labels = [
        format_prompt_label(prompt)
        for prompt in prompt_names
    ]

    prompt_legend = ax.legend(
        legend_handles,
        legend_labels,
        title="Promptvariation",
        loc="upper right",
    )
    ax.add_artist(prompt_legend)

    # Second legend: explanation of boxplot elements.
    boxplot_legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Mittelwert",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Box: mittlere 50 % (Q1–Q3)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1,
            marker="_",
            markersize=8,
            label="Whisker: Werte innerhalb 1,5 × IQR",
        ),
    ]

    ax.legend(
        handles=boxplot_legend_handles,
        title="Boxplot",
        loc="upper left",
    )

    plt.tight_layout()

    output_file = input_file.with_suffix(".png")

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{input_file.name}: Boxplot gespeichert unter: {output_file}"
    )


def get_input_files():
    """
    Return only the Synth Phase 1 and Synth Phase 2 CSV files.

    Command-line arguments take priority. Without arguments, the script
    automatically searches for Synth_Phase_1.csv and Synth_Phase_2.csv in
    the script directory.
    """
    if len(sys.argv) > 1:
        files = []

        for argument in sys.argv[1:]:
            path = Path(argument)

            if not path.is_absolute():
                path = BASE_DIR / path

            if re.search(r"Synth_Phase_[12]", path.stem):
                files.append(path)

        return files

    synth_files = []

    for phase in (1, 2):
        exact_path = BASE_DIR / f"Synth_Phase_{phase}.csv"

        if exact_path.exists():
            synth_files.append(exact_path)

    return synth_files


def main():
    input_files = get_input_files()

    if not input_files:
        raise FileNotFoundError(
            f"No CSV files found in {BASE_DIR}"
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
