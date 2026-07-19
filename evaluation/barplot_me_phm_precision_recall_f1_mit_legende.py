from pathlib import Path
import re

import matplotlib.pyplot as plt


# ==========================================================
# Konfiguration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FOLDERS = [
    BASE_DIR / "ME_Output_Evaluation",
    BASE_DIR / "PHM_Output_Evaluation",
]


# ==========================================================
# Hilfsfunktionen
# ==========================================================

def extract_w_k(filename):
    """
    Liest w und k aus Dateinamen wie:
    evaluation_output_step1_w20_k10.txt

    Rückgabe:
        (20, 10)
    """
    match = re.search(r"_w(\d+)_k(\d+)", filename)

    if not match:
        return None, None

    return int(match.group(1)), int(match.group(2))


def extract_metrics(file_path):
    """
    Liest Precision, Recall und F1-Score aus einer Evaluationsdatei.

    Erwartete Zeilen z. B.:
        Precision: 0.1176
        Recall:    0.7674
        F1 score:  0.2040
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")

    patterns = {
        "Precision": r"Precision\s*[:=]\s*([0-9]*\.?[0-9]+)",
        "Recall": r"Recall\s*[:=]\s*([0-9]*\.?[0-9]+)",
        "F1-Score": r"F1[\s_-]*score\s*[:=]\s*([0-9]*\.?[0-9]+)",
    }

    metrics = {}

    for metric_name, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            raise ValueError(
                f"{metric_name} nicht in Datei gefunden: {file_path.name}"
            )

        metrics[metric_name] = float(match.group(1))

    return metrics


def file_sort_key(file_path):
    """
    Sortiert Dateien numerisch nach w und anschließend nach k.
    Dateien ohne erkennbares w/k-Muster kommen ans Ende.
    """
    w, k = extract_w_k(file_path.name)

    if w is None or k is None:
        return (999999, 999999, file_path.name)

    return (w, k, file_path.name)


def make_label(file_path):
    """
    Erzeugt kompakte X-Achsenbeschriftungen wie:
        w20 / k10
    """
    w, k = extract_w_k(file_path.name)

    if w is None or k is None:
        return file_path.stem

    return f"w{w}\nk{k}"


# ==========================================================
# Plot
# ==========================================================

def create_folder_plot(folder):
    """
    Geht alle .txt-Dateien eines Ordners durch und erstellt
    EIN gemeinsames Balkendiagramm mit Precision, Recall und F1-Score
    für jede TextTiling-Parameterkombination.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Ordner nicht gefunden: {folder}"
        )

    txt_files = sorted(
        folder.glob("*.txt"),
        key=file_sort_key,
    )

    if not txt_files:
        raise FileNotFoundError(
            f"Keine .txt-Dateien in {folder.name} gefunden."
        )

    labels = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for file_path in txt_files:
        try:
            metrics = extract_metrics(file_path)
        except ValueError as exc:
            print(f"Übersprungen: {exc}")
            continue

        labels.append(make_label(file_path))
        precision_scores.append(metrics["Precision"])
        recall_scores.append(metrics["Recall"])
        f1_scores.append(metrics["F1-Score"])

        print(
            f"{folder.name}: {file_path.name} -> "
            f"Precision = {metrics['Precision']:.4f}, "
            f"Recall = {metrics['Recall']:.4f}, "
            f"F1 = {metrics['F1-Score']:.4f}"
        )

    if not f1_scores:
        raise ValueError(
            f"In {folder.name} konnten keine Metriken gelesen werden."
        )

    x_positions = list(range(len(labels)))
    bar_width = 0.25

    # Breite automatisch an die Anzahl der Gruppen anpassen
    figure_width = max(12, len(labels) * 0.9)

    fig, ax = plt.subplots(
        figsize=(figure_width, 7)
    )

    bars_precision = ax.bar(
        [x - bar_width for x in x_positions],
        precision_scores,
        width=bar_width,
        label="Precision",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
    )

    bars_recall = ax.bar(
        x_positions,
        recall_scores,
        width=bar_width,
        label="Recall",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        hatch="//",
    )

    bars_f1 = ax.bar(
        [x + bar_width for x in x_positions],
        f1_scores,
        width=bar_width,
        label="F1-Score",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        hatch="..",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
    )

    ax.set_xlabel("TextTiling-Parameter")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Precision, Recall und F1-Score aller Evaluationen – {folder.name}"
    )

    # Legende/Erklärung der TextTiling-Parameter.
    parameter_legend = (
        "TextTiling-Parameter:\n"
        "w = Token-Sequenzgröße (Pseudosatzgröße)\n"
        "k = Blockgröße (Anzahl der Token-Sequenzen pro Vergleichsblock)"
    )
    ax.text(
        0.99,
        0.72,
        parameter_legend,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    # Einheitliche Y-Achse:
    # 0 bis 1,0 in 0,05-Schritten, damit auch Recall-Werte > 0,5 sichtbar sind.
    upper_limit = 1.0
    ax.set_ylim(0, upper_limit)
    ax.set_yticks([i * 0.05 for i in range(21)])

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.legend(
        title="Metrik",
        loc="upper right",
    )

    # Exakte Werte über die Balken schreiben.
    offset = 0.012

    for bars in (bars_precision, bars_recall, bars_f1):
        for bar in bars:
            score = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + offset,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.tight_layout()

    output_file = BASE_DIR / f"{folder.name}_Precision_Recall_F1_barplot.png"

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nGrafik gespeichert: {output_file}\n"
    )


# ==========================================================
# Main
# ==========================================================

def main():
    errors = []

    for folder in INPUT_FOLDERS:
        try:
            create_folder_plot(folder)
        except Exception as exc:
            errors.append((folder.name, str(exc)))
            print(
                f"FEHLER bei {folder.name}: {exc}"
            )

    if errors:
        print("\nFolgende Ordner konnten nicht vollständig verarbeitet werden:")

        for folder_name, error in errors:
            print(
                f"  - {folder_name}: {error}"
            )


if __name__ == "__main__":
    main()
