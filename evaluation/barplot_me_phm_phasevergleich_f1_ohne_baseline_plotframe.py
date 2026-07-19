from pathlib import Path
import re

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

# Nur Promptfamilien, die tatsächlich zwischen Phase 1 und Phase 2
# verglichen werden. Baseline und Plot Frame werden ausgeschlossen.
PROMPT_ORDER = [
    "Protagonist Frame",
    "Situation Model",
    "Event Frame",
    "Goal Frame",
]


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
        Prompt_11_Protagonist_Frame  -> Protagonist Frame
        Prompt_11a_Protagonist_Frame -> Protagonist Frame
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


def load_phase_data(path):
    """
    Liest prompt_name und f1_score ein und ordnet die Prompts
    ihren gemeinsamen Promptfamilien zu.
    """
    df = read_csv_robustly(path)

    required_columns = {"prompt_name", "f1_score"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name}: Fehlende Spalten: "
            + ", ".join(sorted(missing_columns))
        )

    df["prompt_name"] = df["prompt_name"].astype(str).str.strip()
    df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")
    df = df.dropna(subset=["prompt_name", "f1_score"]).copy()

    df["prompt_family"] = df["prompt_name"].apply(get_prompt_family)

    # Falls dieselbe Familie mehrfach vorkommt, wird der Mittelwert genutzt.
    return (
        df.groupby("prompt_family", as_index=False)["f1_score"]
        .mean()
        .set_index("prompt_family")["f1_score"]
        .to_dict()
    )


# ==========================================================
# Plot
# ==========================================================

def create_comparison_plot(dataset_name, phase_files):
    """
    Erstellt für ME bzw. PHM genau eine Grafik.

    Gleiche Promptfamilien aus Phase 1 und Phase 2 stehen direkt
    nebeneinander. Prompts, die nur in einer Phase vorkommen,
    erhalten nur einen Balken.
    """
    phase1_data = load_phase_data(phase_files["Phase 1"])
    phase2_data = load_phase_data(phase_files["Phase 2"])

    # Nur tatsächlich vorhandene Promptfamilien anzeigen,
    # aber in einer festen methodisch sinnvollen Reihenfolge.
    prompt_families = [
        family
        for family in PROMPT_ORDER
        if family in phase1_data or family in phase2_data
    ]

    x_positions = list(range(len(prompt_families)))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(13, 7))

    phase1_x = []
    phase1_values = []
    phase2_x = []
    phase2_values = []

    for x, family in zip(x_positions, prompt_families):
        if family in phase1_data:
            phase1_x.append(x - bar_width / 2)
            phase1_values.append(phase1_data[family])

        if family in phase2_data:
            phase2_x.append(x + bar_width / 2)
            phase2_values.append(phase2_data[family])

    bars_phase1 = ax.bar(
        phase1_x,
        phase1_values,
        width=bar_width,
        label="Phase 1",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.80,
    )

    bars_phase2 = ax.bar(
        phase2_x,
        phase2_values,
        width=bar_width,
        label="Phase 2",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.80,
        hatch="//",
    )

    # Einheitliche Y-Achse: 0 bis 0,5 in 0,05-Schritten.
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
    ax.set_ylabel("F1-Score")
    ax.set_title(
        f"F1-Score der Promptvarianten – {dataset_name}: Phase 1 vs. Phase 2"
    )

    ax.legend(
        title="Versuchsphase",
        loc="upper right",
    )

    # Exakte F1-Werte über den Balken.
    for bars in (bars_phase1, bars_phase2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    output_file = BASE_DIR / f"{dataset_name}_Phase_1_vs_Phase_2_F1_barplot.png"

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"{dataset_name}: Grafik gespeichert unter {output_file}")


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
            create_comparison_plot(dataset_name, phase_files)
        except Exception as exc:
            errors.append((dataset_name, str(exc)))

    if errors:
        print("\nFolgende Datensätze konnten nicht verarbeitet werden:")

        for dataset_name, error in errors:
            print(f"  - {dataset_name}: {error}")


if __name__ == "__main__":
    main()
