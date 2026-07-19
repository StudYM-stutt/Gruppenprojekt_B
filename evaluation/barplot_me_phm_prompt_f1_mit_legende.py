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


def extract_f1_score(file_path):
    """
    Liest den F1-Wert aus einer Evaluationsdatei.

    Erwartete Zeile z. B.:
        F1 score:  0.2040

    Akzeptiert zusätzlich Varianten wie:
        F1-Score: 0.2040
        F1 Score = 0.2040
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")

    match = re.search(
        r"F1[\s_-]*score\s*[:=]\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Kein F1-Score in Datei gefunden: {file_path.name}"
        )

    return float(match.group(1))


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
    EIN gemeinsames Balkendiagramm mit allen gefundenen F1-Scores.
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
    f1_scores = []

    for file_path in txt_files:
        try:
            f1_score = extract_f1_score(file_path)
        except ValueError as exc:
            print(f"Übersprungen: {exc}")
            continue

        labels.append(make_label(file_path))
        f1_scores.append(f1_score)

        print(
            f"{folder.name}: {file_path.name} -> F1 = {f1_score:.4f}"
        )

    if not f1_scores:
        raise ValueError(
            f"In {folder.name} konnten keine F1-Scores gelesen werden."
        )

    x_positions = range(len(f1_scores))

    # Breite automatisch an die Anzahl der Balken anpassen
    figure_width = max(12, len(f1_scores) * 0.65)

    fig, ax = plt.subplots(
        figsize=(figure_width, 7)
    )

    bars = ax.bar(
        x_positions,
        f1_scores,
        width=0.7,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
    )

    ax.set_xlabel("TextTiling-Parameter")
    ax.set_ylabel("F1-Score")
    ax.set_title(
        f"F1-Scores aller Evaluationen – {folder.name}"
    )

    # Legende/Erklärung der TextTiling-Parameter.
    # w = Token-Sequenzgröße (Pseudosatzgröße)
    # k = Blockgröße: Anzahl der Token-Sequenzen pro Vergleichsblock
    parameter_legend = (
        "TextTiling-Parameter:\n"
        "w = Token-Sequenzgröße (Pseudosatzgröße)\n"
        "k = Blockgröße (Anzahl der Token-Sequenzen pro Vergleichsblock)"
    )
    ax.text(
        0.99,
        0.97,
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

    # Einheitliche Y-Achse für beide Grafiken:
    # 0 bis 0,5 in festen 0,05-Schritten.
    upper_limit = 0.5
    ax.set_ylim(0, 0.5)
    ax.set_yticks([i * 0.05 for i in range(11)])

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    # Exakte F1-Werte über die Balken schreiben
    offset = upper_limit * 0.015

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = BASE_DIR / f"{folder.name}_F1_barplot.png"

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
