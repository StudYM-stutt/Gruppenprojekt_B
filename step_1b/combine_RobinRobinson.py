"""
combine_RobinRobinson.py
============================================================
Creates synthetic texts from Robinson Crusoe and Robin Hood
using a parallel-narrative method:

  - Starts with story A (Robin Hood), reads paragraph by paragraph
  - After each paragraph, there is a SWITCH_PROB chance (default 5%)
    of switching to the other story
  - When switching, continues from where the other story left off
  - This mimics a novel with two parallel narrative threads

Produces 20 variants (seed=1..20), each in its own subfolder:
  Output_Synth/variant_01/ ... variant_20/

Each variant contains:
  combined_text.txt       — clean text for LLM
  combined_readable.txt   — labelled version for inspection
  paragraph_sources.json  — metadata per paragraph
  g_grenzen.txt           — gold standard (Step 4 compatible)
  sentences_numbered.txt  — [N] format for LLM batches
  batches/batch_1/2/3.txt — split into 3 batches with overlap
============================================================
"""

import re
import json
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
PFAD_CRUSOE   = Path(__file__).parent / "RobinsonCrusoe.txt"
PFAD_ROBIN    = Path(__file__).parent / "robinHood.txt"
OUTPUT_DIR    = Path(__file__).parent / "Output_Synth"
MIN_LAENGE    = 150       # minimum paragraph length in characters
SWITCH_PROB   = 0.05      # 5% chance of switching to the other story
N_VARIANTS    = 20        # number of variants to produce
NUM_BATCHES   = 3
BATCH_OVERLAP = 4         # sentences shared between consecutive batches


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_file(path):
    """Reads a text file with encoding fallback."""
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file: {path}")


def clean_gutenberg(text):
    """Removes Project Gutenberg header and footer if present."""
    start = text.find("*** START OF")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def split_paragraphs(text, min_length=MIN_LAENGE):
    """Splits text at blank lines into paragraphs, filters too-short ones."""
    parts = re.split(r'\n\s*\n', text)
    paragraphs = []
    for part in parts:
        normalized = " ".join(part.split())
        if len(normalized) >= min_length:
            paragraphs.append(normalized)
    return paragraphs


# ============================================================
# PARALLEL CORPUS BUILDER  
# ============================================================

def create_parallel_corpus(paras_a, source_a, paras_b, source_b,
                           switch_prob, rng):
    """
    Builds a mixed text by reading paragraph by paragraph from story A,
    with a switch_prob chance after each paragraph of switching to story B.
    When switching, continues from where the other story left off.

    Both stories are read from the beginning (index 0).
    Stops when either story runs out of paragraphs.

    Returns:
        entries       : list of {paragraph_nr, source, orig_idx, text}
        gold_boundaries: paragraph_nr values where the source changes
    """
    idx_a = 0   # current position in story A
    idx_b = 0   # current position in story B

    current_source = source_a   # start with story A
    entries        = []
    gold           = []

    while idx_a < len(paras_a) and idx_b < len(paras_b):
        # pick the paragraph from the currently active story
        if current_source == source_a:
            text     = paras_a[idx_a]
            orig_idx = idx_a
            idx_a   += 1
        else:
            text     = paras_b[idx_b]
            orig_idx = idx_b
            idx_b   += 1

        para_nr = len(entries)

        # record source switch as a gold boundary
        if para_nr > 0 and current_source != entries[-1]["source"]:
            gold.append(para_nr)

        entries.append({
            "paragraph_nr": para_nr,
            "source":       current_source,
            "orig_idx":     orig_idx,
            "text":         text,
        })

        # after each paragraph: roll the dice
        if rng.random() < switch_prob:
            if current_source == source_a:
                current_source = source_b
            else:
                current_source = source_a

    return entries, gold


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def sentences_from_entries(entries):
    """
    Splits paragraphs into sentences using punctuation.
    Returns (sentences, source_per_sent, sent_boundaries).
    """
    sentences       = []
    source_per_sent = []

    for e in entries:
        sents = re.split(r'(?<=[.!?])[ ]+(?=[A-Z"])', e["text"])
        for s in sents:
            s = s.strip()
            if s:
                sentences.append(s)
                source_per_sent.append(e["source"])

    boundaries = [
        i for i in range(1, len(source_per_sent))
        if source_per_sent[i] != source_per_sent[i - 1]
    ]
    return sentences, source_per_sent, boundaries


# ============================================================
# BATCH SPLITTER
# ============================================================

def make_batches(total, num_batches, overlap):
    """Splits total sentences into num_batches with overlap."""
    step = (total - overlap) // num_batches
    batches = []
    for i in range(num_batches):
        start = i * step
        end   = total - 1 if i == num_batches - 1 else start + step + overlap - 1
        batches.append({"number": i + 1, "start": start, "end": end,
                        "count": end - start + 1})
    return batches


# ============================================================
# FILE WRITER
# ============================================================

def write_variant(out_dir, entries, gold_para, sentences,
                  source_per_sent, sent_boundaries, batches_meta, seed):
    """Writes all output files for one variant."""
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = out_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    # combined_text.txt
    (out_dir / "combined_text.txt").write_text(
        "\n\n".join(e["text"] for e in entries), encoding="utf-8")

    # combined_readable.txt
    lines = []
    for e in entries:
        lines.append(f"[PARAGRAPH {e['paragraph_nr']} | "
                     f"Source: {e['source']} | "
                     f"Original index: {e['orig_idx']}]")
        lines.append(e["text"] + "\n")
    (out_dir / "combined_readable.txt").write_text(
        "\n".join(lines), encoding="utf-8")

    # paragraph_sources.json
    switches = sum(1 for i in range(1, len(entries))
                   if entries[i]["source"] != entries[i - 1]["source"])
    meta = {
        "seed":             seed,
        "switch_prob":      SWITCH_PROB,
        "num_paragraphs":   len(entries),
        "source_switches":  switches,
        "sources":          [entries[0]["source"],
                             next(e["source"] for e in entries
                                  if e["source"] != entries[0]["source"])],
        "paragraphen_info": [{"paragraph_nr": e["paragraph_nr"],
                               "quelle":        e["source"],
                               "original_index": e["orig_idx"]}
                              for e in entries],
    }
    (out_dir / "paragraph_sources.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # g_grenzen.txt  — Step 4 compatible
    grenzen = [str(len(sent_boundaries))] + [str(b) for b in sent_boundaries]
    (out_dir / "g_grenzen.txt").write_text(
        "\n".join(grenzen), encoding="utf-8")

    # sentences_numbered.txt  — [N] format, Step 4 compatible
    (out_dir / "sentences_numbered.txt").write_text(
        "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sentences)),
        encoding="utf-8")

    # goldstandard.json
    gold = {
        "seed":            seed,
        "switch_prob":     SWITCH_PROB,
        "level":           "sentence",
        "num_sentences":   len(sentences),
        "num_boundaries":  len(sent_boundaries),
        "boundaries":      sent_boundaries,
        "paragraph_gold":  gold_para,
        "source_per_sent": source_per_sent,
    }
    (out_dir / "goldstandard.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

    # batches
    for b in batches_meta:
        chunk = sentences[b["start"]: b["end"] + 1]
        text  = "\n\n".join(f"[{b['start'] + j}] {s}"
                            for j, s in enumerate(chunk))
        (batch_dir / f"batch_{b['number']}.txt").write_text(
            text + "\n", encoding="utf-8")

    (batch_dir / "batches_metadata.json").write_text(
        json.dumps({"num_batches": len(batches_meta),
                    "overlap":     BATCH_OVERLAP,
                    "total_sentences": len(sentences),
                    "batches":     batches_meta},
                   ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("COMBINE: Robin Hood + Robinson Crusoe")
    print(f"  Method     : parallel narrative ({SWITCH_PROB*100:.0f}% switch probability)")
    print(f"  Variants   : {N_VARIANTS}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 60)

    # --- Read and clean ---
    print("\nReading files ...")
    robin_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_ROBIN)))
    crusoe_paras = split_paragraphs(clean_gutenberg(read_file(PFAD_CRUSOE)))
    print(f"  Robin Hood       : {len(robin_paras)} paragraphs")
    print(f"  Robinson Crusoe  : {len(crusoe_paras)} paragraphs")

    # --- Generate variants ---
    for seed in range(1, N_VARIANTS + 1):
        var_dir = OUTPUT_DIR / f"variant_{seed:02d}"
        rng = random.Random(seed)

        entries, gold_para = create_parallel_corpus(
            robin_paras,  "Robin_Hood",
            crusoe_paras, "Robinson_Crusoe",
            SWITCH_PROB, rng
        )

        sentences, src_sent, s_gold = sentences_from_entries(entries)
        batches = make_batches(len(sentences), NUM_BATCHES, BATCH_OVERLAP)
        write_variant(var_dir, entries, gold_para,
                      sentences, src_sent, s_gold, batches, seed)

        switches = sum(1 for i in range(1, len(entries))
                       if entries[i]["source"] != entries[i - 1]["source"])
        print(f"  [{seed:02d}/{N_VARIANTS}] seed={seed:2d} | "
              f"{len(entries):4d} paras | {switches:3d} switches | "
              f"{len(sentences):5d} sents | {len(s_gold):3d} boundaries")

    print("\n" + "=" * 60)
    print(f"DONE — {N_VARIANTS} variants in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
