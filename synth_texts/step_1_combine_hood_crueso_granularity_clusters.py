"""
step_1_combine_hood_crueso_granularity_clusters.py
============================================================
Creates 20 shorter synthetic paragraph-level texts from Robin Hood and
Robinson Crusoe for an LLM-based narrative segmentation project.

Main idea
---------
The 20 variants are divided into 4 experimental granularity clusters:

  Cluster 1 | variant_01-variant_05 | coarse mixing      | long source blocks
  Cluster 2 | variant_06-variant_10 | medium mixing      | medium source blocks
  Cluster 3 | variant_11-variant_15 | fine mixing        | short source blocks
  Cluster 4 | variant_16-variant_20 | very fine mixing   | very short source blocks

Each variant is deliberately shortened with TARGET_WORDS and MAX_PARAGRAPHS,
so Step 4 creates fewer LLM batches and is less likely to hit token limits.

Output per variant
------------------
  paragraphs_numbered.txt      required by Step 4, [N] paragraph format
  g_grenzen_paragraph.txt      gold boundaries where source changes
  source_map.tsv               debugging/evaluation helper
  variant_metadata.txt         cluster settings and summary

Important
---------
Gold boundaries are paragraph numbers at which the source changes. Example:
if paragraph 25 is the first paragraph from the other source, then 25 is a
boundary.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent

PFAD_ROBIN = SCRIPT_DIR / "step_1_input" / "robin_hood.txt"
PFAD_CRUSOE = SCRIPT_DIR / "step_1_input" / "robinson_crusoe.txt"

OUTPUT_DIR = SCRIPT_DIR / "step_1_output"

N_VARIANTS = 20
MIN_PARAGRAPH_CHARS = 150

# Makes the synthetic texts much shorter than full novels.
# Adjust these two values if Step 4 is still too slow.
TARGET_WORDS = 30_000
MAX_PARAGRAPHS = 500
MIN_PARAGRAPHS = 45

# Optional: avoid always starting at the beginning of both novels.
# This gives the variants more diversity while preserving internal paragraph order.
RANDOM_START_WINDOWS = True
MAX_START_OFFSET_SHARE = 0.55

SOURCE_A = "Robin_Hood"
SOURCE_B = "Robinson_Crusoe"


@dataclass(frozen=True)
class ClusterConfig:
    cluster_id: int
    label: str
    variant_start: int
    variant_end: int
    run_min: int
    run_max: int
    target_words: int = TARGET_WORDS
    max_paragraphs: int = MAX_PARAGRAPHS


CLUSTERS: list[ClusterConfig] = [
    # Long blocks: few boundaries, coarse segmentation problem.
    ClusterConfig(1, "coarse_long_blocks", 1, 5, run_min=18, run_max=32),
    # Medium blocks: balanced source switches.
    ClusterConfig(2, "medium_blocks", 6, 10, run_min=9, run_max=17),
    # Short blocks: more frequent source switches.
    ClusterConfig(3, "fine_short_blocks", 11, 15, run_min=4, run_max=8),
    # Very short blocks: many boundaries, hardest / most granular condition.
    ClusterConfig(4, "very_fine_micro_blocks", 16, 20, run_min=1, run_max=3),
]


class Entry(TypedDict):
    paragraph_nr: int
    source: str
    orig_idx: int
    text: str
    word_count: int


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


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def split_paragraphs(text: str, min_length: int = MIN_PARAGRAPH_CHARS) -> list[str]:
    """Splits text at blank lines into paragraphs and filters very short paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []

    for part in parts:
        normalized = normalize_whitespace(part)
        if len(normalized) >= min_length:
            paragraphs.append(normalized)

    return paragraphs


def get_cluster_for_variant(variant_nr: int) -> ClusterConfig:
    for cluster in CLUSTERS:
        if cluster.variant_start <= variant_nr <= cluster.variant_end:
            return cluster
    raise ValueError(f"No cluster configured for variant {variant_nr}")


def choose_start_index(paragraphs: list[str], rng: random.Random) -> int:
    """
    Chooses a start index inside the first part of a text.
    Keeps enough remaining material while avoiding identical beginnings.
    """
    if not RANDOM_START_WINDOWS:
        return 0
    max_start = int(len(paragraphs) * MAX_START_OFFSET_SHARE)
    max_start = max(0, min(max_start, len(paragraphs) - MIN_PARAGRAPHS - 1))
    return rng.randint(0, max_start) if max_start > 0 else 0


# ============================================================
# SYNTHETIC CORPUS BUILDER
# ============================================================


def create_mixed_corpus(
    paras_a: list[str],
    source_a: str,
    paras_b: list[str],
    source_b: str,
    cluster: ClusterConfig,
    rng: random.Random,
) -> tuple[list[Entry], list[int]]:
    """
    Builds a shorter mixed text by alternating source blocks.

    The block length is sampled from cluster.run_min to cluster.run_max.
    This creates controlled granularity categories instead of one global
    SWITCH_PROB. The text stops when target_words or max_paragraphs is reached.
    """
    idx_a = choose_start_index(paras_a, rng)
    idx_b = choose_start_index(paras_b, rng)

    current_source = rng.choice([source_a, source_b])
    entries: list[Entry] = []
    gold: list[int] = []
    total_words = 0

    while (
        total_words < cluster.target_words
        and len(entries) < cluster.max_paragraphs
        and idx_a < len(paras_a)
        and idx_b < len(paras_b)
    ):
        run_length = rng.randint(cluster.run_min, cluster.run_max)

        for _ in range(run_length):
            if total_words >= cluster.target_words or len(entries) >= cluster.max_paragraphs:
                break

            if current_source == source_a:
                if idx_a >= len(paras_a):
                    break
                text = paras_a[idx_a]
                orig_idx = idx_a
                idx_a += 1
            else:
                if idx_b >= len(paras_b):
                    break
                text = paras_b[idx_b]
                orig_idx = idx_b
                idx_b += 1

            para_nr = len(entries)
            if para_nr > 0 and current_source != entries[-1]["source"]:
                gold.append(para_nr)

            wc = word_count(text)
            entries.append(
                {
                    "paragraph_nr": para_nr,
                    "source": current_source,
                    "orig_idx": orig_idx,
                    "text": text,
                    "word_count": wc,
                }
            )
            total_words += wc

        current_source = source_b if current_source == source_a else source_a

    if len(entries) < MIN_PARAGRAPHS:
        raise RuntimeError(
            f"Variant too short: only {len(entries)} paragraphs. "
            f"Lower MIN_PARAGRAPHS or check input files."
        )

    return entries, gold


# ============================================================
# FILE WRITER
# ============================================================


def write_variant(
    out_dir: Path,
    entries: list[Entry],
    gold_para: list[int],
    cluster: ClusterConfig,
    seed: int,
) -> None:
    """Writes Step 4 files plus transparent metadata/debug files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "paragraphs_numbered.txt").write_text(
        "\n\n".join(f"[{e['paragraph_nr']}] {e['text']}" for e in entries),
        encoding="utf-8",
    )

    paragraph_boundaries = [str(len(gold_para))] + [str(boundary) for boundary in gold_para]
    (out_dir / "g_grenzen_paragraph.txt").write_text(
        "\n".join(paragraph_boundaries),
        encoding="utf-8",
    )

    (out_dir / "source_map.tsv").write_text(
        "paragraph_nr\tsource\torig_idx\tword_count\n"
        + "\n".join(
            f"{e['paragraph_nr']}\t{e['source']}\t{e['orig_idx']}\t{e['word_count']}"
            for e in entries
        ),
        encoding="utf-8",
    )

    total_words = sum(e["word_count"] for e in entries)
    (out_dir / "variant_metadata.txt").write_text(
        "\n".join(
            [
                f"variant_seed={seed}",
                f"cluster_id={cluster.cluster_id}",
                f"cluster_label={cluster.label}",
                f"run_length_min={cluster.run_min}",
                f"run_length_max={cluster.run_max}",
                f"target_words={cluster.target_words}",
                f"max_paragraphs={cluster.max_paragraphs}",
                f"actual_paragraphs={len(entries)}",
                f"actual_words={total_words}",
                f"gold_boundaries={len(gold_para)}",
                f"boundary_list={','.join(map(str, gold_para))}",
            ]
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("=" * 70)
    print("COMBINE: Robin Hood + Robinson Crusoe")
    print("Method   : controlled granularity clusters, paragraph-level gold")
    print(f"Variants : {N_VARIANTS} total = 4 clusters x 5 variants")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Length   : target ~{TARGET_WORDS} words, max {MAX_PARAGRAPHS} paragraphs")
    print("Files    : paragraphs_numbered.txt, g_grenzen_paragraph.txt")
    print("=" * 70)

    print("\nReading files ...")
    robin_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_ROBIN)))
    crusoe_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_CRUSOE)))

    print(f"  Robin Hood       : {len(robin_paras)} paragraphs")
    print(f"  Robinson Crusoe  : {len(crusoe_paras)} paragraphs")

    for variant_nr in range(1, N_VARIANTS + 1):
        cluster = get_cluster_for_variant(variant_nr)
        var_dir = OUTPUT_DIR / f"variant_{variant_nr:02d}"
        rng = random.Random(variant_nr)

        entries, gold_para = create_mixed_corpus(
            robin_paras,
            SOURCE_A,
            crusoe_paras,
            SOURCE_B,
            cluster,
            rng,
        )

        write_variant(var_dir, entries, gold_para, cluster, seed=variant_nr)

        total_words = sum(e["word_count"] for e in entries)
        print(
            f"  [{variant_nr:02d}/{N_VARIANTS}] "
            f"cluster={cluster.cluster_id} {cluster.label:23s} | "
            f"paras={len(entries):3d} | words={total_words:5d} | "
            f"boundaries={len(gold_para):3d} | run={cluster.run_min}-{cluster.run_max}"
        )

    print("\n" + "=" * 70)
    print(f"DONE — {N_VARIANTS} variants written to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
