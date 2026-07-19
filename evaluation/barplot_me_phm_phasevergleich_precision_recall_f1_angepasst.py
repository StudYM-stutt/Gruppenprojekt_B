from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Konfiguration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "ME": {
        "Phase 1": BASE_DIR / "ME_Phase_1.csv",
        "Phase 2": BASE_DIR / "ME_Phase_2.csv",
    },
    "PHM": {
        "Phase 1": BASE_DIR / "PHM_Phase_1.csv",
        "Phase 2": BASE_DIR / "PHM_Phase_2.csv",
    },
}

# Baseline und Plot Frame werden weiterhin nicht verglichen.
PROMPT_ORDER = [
    "Protagonist Frame",
    "Situation Model",
    "Event Frame",
    "Goal Frame",
]

METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-Score"),
]

# Einheitliche Farben je Metrik über beide Phasen hinweg.
METRIC_COLORS = {
    "precision": "tab:blue",
    "recall": "tab:orange",
    "f1_score": "tab:green",
}


# ==========================================================
# Hilfsfunktionen
# ==========================================================

def read_csv_robustly(path):
    """
    Liest komma-, semikolon- oder tabulatorgetrennte CSV-Dateien.
    """
    return pd.read_csv(path, sep=None, engine="python")


def get_prompt_family(prompt_name):
    """
    Ordnet Phase-1- und Phase-2-Prompts derselben Promptfamilie zu.

    Beispiele:
        Prompt_11_Protagonist_Frame   -> Protagonist Frame
        Prompt_11a_Protagonist_Frame  -> Protagonist Frame
    """
    name = str(prompt_name).lower()

    if "baseline" in name:
        return "Baseline"
    if "plot_frame" in name:
        return "Plot Frame"
    if "protagonist_frame" in name:
        return "Protagonist Frame"
    if "situation_model" in name:
        return "Situation Model"
    if "event_frame" in name:
        return "Event Frame"
    if "goal_frame" in name:
        return "Goal Frame"

    return str(prompt_name)


def resolve_metric_columns(df):
    """
    Erkennt gebräuchliche Spaltennamen für Precision, Recall und F1.
    """
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    alternatives = {
        "precision": ["precision", "precision_score"],
        "recall": ["recall", "recall_score"],
        "f1_score": ["f1_score", "f1", "f1-score"],
    }

    resolved = {}

    for canonical_name, candidates in alternatives.items():
        for candidate in candidates:
            if candidate in lookup:
                resolved[canonical_name] = lookup[candidate]
                break

        if canonical_name not in resolved:
            raise ValueError(
                f"Keine passende Spalte für '{canonical_name}' gefunden."
            )

    return resolved


def load_phase_data(path):
    """
    Liest Precision, Recall und F1-Score je Promptfamilie ein.

    Falls dieselbe Promptfamilie mehrfach vorkommt, wird für jede
    Metrik der Mittelwert gebildet.
    """
    df = read_csv_robustly(path)

    if "prompt_name" not in df.columns:
        raise ValueError(
            f"{path.name}: Fehlende Spalte 'prompt_name'."
        )

    metric_columns = resolve_metric_columns(df)

    df["prompt_name"] = df["prompt_name"].astype(str).str.strip()
    df["prompt_family"] = df["prompt_name"].apply(get_prompt_family)

    for canonical_name, actual_column in metric_columns.items():
        df[canonical_name] = pd.to_numeric(
            df[actual_column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["precision", "recall", "f1_score"],
        how="all",
    ).copy()

    grouped = (
        df.groupby("prompt_family", as_index=True)[
            ["precision", "recall", "f1_score"]
        ]
        .mean()
    )

    return grouped.to_dict(orient="index")


# ==========================================================
# Plot
# ==========================================================

def create_comparison_plot(dataset_name, phase_files):
    """
    Erstellt für ME bzw. PHM jeweils EINE gemeinsame Grafik.

    Für jede Promptfamilie werden Precision, Recall und F1-Score
    für Phase 1 und Phase 2 direkt miteinander verglichen.

    Baseline und Plot Frame werden nicht dargestellt.
    """
    phase1_data = load_phase_data(phase_files["Phase 1"])
    phase2_data = load_phase_data(phase_files["Phase 2"])

    prompt_families = [
        family
        for family in PROMPT_ORDER
        if family in phase1_data or family in phase2_data
    ]

    if not prompt_families:
        raise ValueError(
            f"{dataset_name}: Keine vergleichbaren Promptfamilien gefunden."
        )

    x_positions = list(range(len(prompt_families)))

    # Sechs Balken pro Promptfamilie:
    # Phase 1: Precision, Recall, F1
    # Phase 2: Precision, Recall, F1
    bar_width = 0.12

    offsets = {
        ("Phase 1", "precision"): -2.5 * bar_width,
        ("Phase 1", "recall"): -1.5 * bar_width,
        ("Phase 1", "f1_score"): -0.5 * bar_width,
        ("Phase 2", "precision"): 0.5 * bar_width,
        ("Phase 2", "recall"): 1.5 * bar_width,
        ("Phase 2", "f1_score"): 2.5 * bar_width,
    }

    fig, ax = plt.subplots(figsize=(15, 8))

    plotted_bars = {}

    for phase_name, phase_data in [
        ("Phase 1", phase1_data),
        ("Phase 2", phase2_data),
    ]:
        for metric_key, metric_label in METRICS:
            xs = []
            values = []

            for x, family in zip(x_positions, prompt_families):
                if family not in phase_data:
                    continue

                value = phase_data[family].get(metric_key)

                if pd.isna(value):
                    continue

                xs.append(
                    x + offsets[(phase_name, metric_key)]
                )
                values.append(value)

            if not values:
                continue

            hatch = None if phase_name == "Phase 1" else "//"

            bars = ax.bar(
                xs,
                values,
                width=bar_width,
                label=f"{phase_name} – {metric_label}",
                color=METRIC_COLORS[metric_key],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.80,
                hatch=hatch,
            )

            plotted_bars[(phase_name, metric_key)] = bars

    # Gemeinsame Skala für Precision, Recall und F1:
    # 0 bis 0,5 in festen Schritten von 0,05.
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([i * 0.05 for i in range(11)])

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(prompt_families)

    ax.set_xlabel("Promptvariation")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Precision, Recall und F1-Score der Promptvarianten – "
        f"{dataset_name}: Phase 1 vs. Phase 2"
    )

    ax.legend(
        title="Phase und Metrik",
        loc="upper right",
        ncol=2,
    )

    # Exakte Werte über den Balken.
    for bars in plotted_bars.values():
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.tight_layout()

    output_file = BASE_DIR / (
        f"{dataset_name}_Phase_1_vs_Phase_2_"
        f"Precision_Recall_F1_barplot.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{dataset_name}: Grafik gespeichert unter {output_file}"
    )


# ==========================================================
# Main
# ==========================================================

def main():
    errors = []

    for dataset_name, phase_files in DATASETS.items():
        missing = [
            path.name
            for path in phase_files.values()
            if not path.exists()
        ]

        if missing:
            errors.append(
                (
                    dataset_name,
                    "Fehlende Datei(en): " + ", ".join(missing),
                )
            )
            continue

        try:
            create_comparison_plot(
                dataset_name,
                phase_files,
            )
        except Exception as exc:
            errors.append(
                (dataset_name, str(exc))
            )

    if errors:
        print(
            "\nFolgende Datensätze konnten nicht verarbeitet werden:"
        )

        for dataset_name, error in errors:
            print(
                f"  - {dataset_name}: {error}"
            )


if __name__ == "__main__":
    main()
