"""
combine_RobinRobinson.py
============================================================
Creates synthetic paragraph-level texts from Robinson Crusoe and Robin Hood
using a parallel-narrative method:

  - Starts with story A (Robin Hood), reads paragraph by paragraph
  - After each paragraph, there is a SWITCH_PROB chance of switching to the other story
  - When switching, continues from where the other story left off
  - This mimics a novel with two parallel narrative threads

Produces 20 variants (seed=1..20), each in its own subfolder:
  step_1_output/variant_01/ ... variant_20/

Each variant contains only the files needed for the paragraph-level Step 4 pipeline:
  paragraphs_numbered.txt    — [N] format on paragraph level for LLM batches
  g_grenzen_paragraph.txt    — paragraph-level gold standard for source switches
============================================================
"""

import re
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent

PFAD_ROBIN = SCRIPT_DIR / "step_1_input" / "robin_hood.txt"
PFAD_CRUSOE = SCRIPT_DIR / "step_1_input" / "robinson_crusoe.txt"

OUTPUT_DIR  = SCRIPT_DIR / "step_1_output"
MIN_LAENGE  = 150       # minimum paragraph length in characters
SWITCH_PROB = 0.05      # 5% chance of switching to the other story
N_VARIANTS  = 20        # number of variants to produce


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_file(path: Path) -> str:
    """Reads a text file with encoding fallback."""
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file: {path}")


def clean_gutenberg(text: str) -> str:
    """Removes Project Gutenberg header and footer if present."""
    start = text.find("*** START OF")
    if start != -1:
        text = text[text.find("\n", start) + 1:]

    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]

    return text.strip()


def split_paragraphs(text: str, min_length: int = MIN_LAENGE) -> list[str]:
    """Splits text at blank lines into paragraphs and filters too-short paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []

    for part in parts:
        normalized = " ".join(part.split())
        if len(normalized) >= min_length:
            paragraphs.append(normalized)

    return paragraphs


# ============================================================
# PARALLEL CORPUS BUILDER
# ============================================================

def create_parallel_corpus(
    paras_a: list[str],
    source_a: str,
    paras_b: list[str],
    source_b: str,
    switch_prob: float,
    rng: random.Random,
) -> tuple[list[dict[str, int | str]], list[int]]:
    """
    Builds a mixed text by reading paragraph by paragraph from story A,
    with a switch_prob chance after each paragraph of switching to story B.
    When switching, continues from where the other story left off.

    Both stories are read from the beginning.
    Stops when either story runs out of paragraphs.

    Returns:
        entries: list of paragraph metadata and text
        gold: paragraph numbers where the source changes
    """
    idx_a = 0
    idx_b = 0
    current_source = source_a

    entries: list[dict[str, int | str]] = []
    gold: list[int] = []

    while idx_a < len(paras_a) and idx_b < len(paras_b):
        if current_source == source_a:
            text = paras_a[idx_a]
            orig_idx = idx_a
            idx_a += 1
        else:
            text = paras_b[idx_b]
            orig_idx = idx_b
            idx_b += 1

        para_nr = len(entries)

        if para_nr > 0 and current_source != entries[-1]["source"]:
            gold.append(para_nr)

        entries.append(
            {
                "paragraph_nr": para_nr,
                "source": current_source,
                "orig_idx": orig_idx,
                "text": text,
            }
        )

        if rng.random() < switch_prob:
            current_source = source_b if current_source == source_a else source_a

    return entries, gold


# ============================================================
# FILE WRITER
# ============================================================

def write_variant(out_dir: Path, entries: list[dict[str, int | str]], gold_para: list[int]) -> None:
    """Writes only the two files needed for the paragraph-level Step 4 pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # paragraphs_numbered.txt — paragraph-level [N] format without source metadata
    (out_dir / "paragraphs_numbered.txt").write_text(
        "\n\n".join(f"[{e['paragraph_nr']}] {e['text']}" for e in entries),
        encoding="utf-8",
    )

    # g_grenzen_paragraph.txt — paragraph-level boundaries where the source changes
    paragraph_boundaries = [str(len(gold_para))] + [str(boundary) for boundary in gold_para]
    (out_dir / "g_grenzen_paragraph.txt").write_text(
        "\n".join(paragraph_boundaries),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("COMBINE: Robin Hood + Robinson Crusoe")
    print(f"  Method     : parallel narrative ({SWITCH_PROB * 100:.0f}% switch probability)")
    print(f"  Variants   : {N_VARIANTS}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("  Files      : paragraphs_numbered.txt, g_grenzen_paragraph.txt")
    print("=" * 60)

    print("\nReading files ...")
    robin_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_ROBIN)))
    crusoe_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_CRUSOE)))

    print(f"  Robin Hood       : {len(robin_paras)} paragraphs")
    print(f"  Robinson Crusoe  : {len(crusoe_paras)} paragraphs")

    for seed in range(1, N_VARIANTS + 1):
        var_dir = OUTPUT_DIR / f"variant_{seed:02d}"
        rng = random.Random(seed)

        entries, gold_para = create_parallel_corpus(
            robin_paras,
            "Robin_Hood",
            crusoe_paras,
            "Robinson_Crusoe",
            SWITCH_PROB,
            rng,
        )

        write_variant(var_dir, entries, gold_para)

        print(
            f"  [{seed:02d}/{N_VARIANTS}] seed={seed:2d} | "
            f"{len(entries):4d} paras | {len(gold_para):3d} paragraph boundaries"
        )

    print("\n" + "=" * 60)
    print(f"DONE — {N_VARIANTS} variants in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
