# Gruppenprojekt B – Text Segmentation with LLMs

## Project Goal

This project investigates whether Large Language Models (LLMs) can identify boundaries between two interleaved texts. The results are compared against the classical TextTiling algorithm (NLTK). The project runs two parallel pipelines: one for real literary texts and one for synthetically generated mixed texts.

---

## Background

At the start, synthetic texts were created in a simple A-B-A-B pattern to test whether LLMs could detect text switches at all. Various text combinations were explored, including Alice in Wonderland, Frankenstein, and Romeo & Juliet. Throughout the project, paragraph-based vs. sentence-based segmentation was also evaluated: sentence-level segmentation was tested first (via `step_4_synth_llm.py`), but paragraph-level segmentation was found to produce more accurate results. The pipeline was restructured to work at the paragraph level (`step_1_combine_hood_crueso_paragraph_minimal.py`, `step_4_synth_llm_dynamic.py`).

To make the synthetic texts comparable to the real literary texts (Project Hail Mary, median paragraph size ~282 words), the local `combine_RobinRobinson.py` merges consecutive same-source paragraphs into windows of approximately `TARGET_WORDS = 250` words. This ensures that the LLM receives similarly sized input chunks across both pipelines.

**Robin Hood + Robinson Crusoe** was selected as the final pairing for synthetic texts due to the clear thematic contrast between the two books.

According to the project meeting, testing **5 variants** is sufficient (instead of 20).

---

## Repository Structure

```
Gruppenprojekt_B/
│
├── main_pipeline/              # Pipeline for real literary texts
│   ├── step_1_input/           # Input: g_Standard.txt (full book text)
│   ├── Grenzen/                # Gold standard: g_grenzen.txt (true chapter boundaries)
│   ├── step_1_output/          # TextTiling segmentation output
│   ├── step_2_output/          # Evaluation results (Precision, Recall, F1)
│   ├── step_3_output/          # Distance visualizations
│   ├── step_4_output/          # LLM segmentation output
│   ├── step_5_output/          # LLM result visualizations
│   ├── prompts/                # Prompt .txt files for LLM
│   ├── prompt_outputs/         # LLM responses per prompt
│   ├── llm_cache/              # Cache for LLM responses
│   ├── llm_cache_step4/        # Cache for step 4 LLM calls
│   ├── step_1_texttiling.py
│   ├── step_2_evaluation_batch_distance.py
│   ├── step_3_visualization_distance.py
│   ├── step_4_llm_dynamic_batches.py
│   ├── step_5_visualisation_llm_grouped.py
│   ├── texttiling_de.py        # German TextTiling implementation
│   └── texttiling_eng.py       # English TextTiling implementation
│
├── synth_texts/                # Pipeline for synthetic mixed texts
│   ├── step_1_input/           # Input: robinHood.txt, RobinsonCrusoe.txt
│   ├── step_1_output/          # Generated variants (variant_01/ … variant_20/)
│   ├── step_4_output/          # LLM segmentation output
│   ├── step_4_prompt_input/    # Prompts for synthetic text LLM
│   ├── step_5_output/          # LLM result visualizations
│   ├── llm_cache_step4/        # Cache for step 4 LLM calls
│   ├── step_1_combine_hood_crueso.py
│   ├── step_1_combine_hood_crueso_paragraph_minimal.py
│   ├── step_4_synth_llm.py
│   ├── step_4_synth_llm_dynamic.py
│   └── step_5_synth_visualisation_llm_grouped.py
│
├── step_1_combine_hood_crueso_granularity_clusters.py  # New: 4 granularity categories
├── README.md
└── requirements.txt
```

---

## Pipeline 1: Main Pipeline (Real Texts)

Used for: **Project Hail Mary**, **Michael Ende – Die unendliche Geschichte**

### Input files

| File | Location | Description |
|------|----------|-------------|
| `g_Standard.txt` | `step_1_input/` | Full book text |
| `g_grenzen.txt` | `Grenzen/` | True chapter/section boundaries (gold standard) |

### Step 1 – TextTiling (`step_1_texttiling.py`)

Segments the text using the TextTiling algorithm from NLTK.

**How to run:**
```bash
python step_1_texttiling.py
```

**Parameters (set at top of file):**

| Parameter | Description |
|-----------|-------------|
| Language | Switch between `texttiling_de` (German) or `texttiling_eng` (English) |
| `w` | Window size — 20 combinations tested, ranging from w=20 to w=200 |
| `k` | Comparison window — always approximately w/2 |

**Output:** `step_1_output/` — predicted segment boundaries

---

### Step 2 – Evaluation (`step_2_evaluation_batch_distance.py`)

Compares TextTiling output against gold standard boundaries. Processes all `output_step1_w*_k*.txt` files from step 1 automatically.

**How to run:**
```bash
python step_2_evaluation_batch_distance.py
```

**Input:** `step_1_output/output_step1_w*_k*.txt` + `Grenzen/g_grenzen.txt`

**Ground truth format:** First line = total number of boundaries; following lines = paragraph IDs where a boundary begins.

**Output:** `step_2_output/evaluation_output_step1_w{w}_k{k}.txt` per parameter combination — TP/FP/FN, Precision, Recall, F1-Score, and distance analysis (exact matches, within ±1/±2/±3/±5 paragraphs)

---

### Step 3 – Visualization (`step_3_visualization_distance.py`)

Reads all evaluation files from `step_2_output/` and creates plots and a summary CSV comparing all parameter combinations.

```bash
python step_3_visualization_distance.py
```

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALUATION_INPUT_DIR` | `step_2_output/` | Directory with step 2 evaluation files |
| `VISUALIZATION_OUTPUT_DIR` | `step_3_output/` | Output directory |
| `EVALUATION_FILE_PATTERN` | `evaluation_output_step1_w*_k*.txt` | File name pattern to match |

**Output** (`step_3_output/`):

| File | Description |
|------|-------------|
| `precision_recall_f1_plot.png` | Precision, Recall, and F1 per parameter combination |
| `distance_matches_plot.png` | Boundary matches at ±1/±2/±3/±5 paragraphs per parameter combination |
| `distance_threshold_curve.png` | Cumulative match curve across distance thresholds |
| `evaluation_results_table.csv` | All metrics per parameter combination as CSV (incl. Jaccard index and match shares) |

---

### Step 4 – LLM Segmentation (`step_4_llm_dynamic_batches.py`)

Sends batches of text to an LLM and asks it to identify segment boundaries.

**Setup:**
```bash
pip3 install openai
# For OpenAI:
export OPENAI_API_KEY="your_key_here"
```

**How to run (OpenAI):**
```bash
python step_4_llm_dynamic_batches.py --provider openai
```

**How to run (dry run for testing – no API key needed):**
```bash
python step_4_llm_dynamic_batches.py --provider dryrun
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--provider` | `dryrun` | LLM provider |
| `--model` | `gpt-4o-mini` | Model to use |
| `--batch-mode` | `words` | `words` (dynamic word-count) or `paragraphs` (fixed count) |
| `--min-batch-words` | `650` | Minimum words per batch (words mode) |
| `--max-batch-words` | `1100` | Maximum words per batch (words mode) |
| `--batch-size` | `40` | Paragraphs per batch (paragraphs mode only) |
| `--dynamic-context` | `True` | Grow context window until min words reached |
| `--context-size` | `3` | Context paragraphs per side (if dynamic context disabled) |
| `--min-context-words` | `180` | Minimum words per side for dynamic context |
| `--max-context-paragraphs` | `12` | Maximum context paragraphs per side |
| `--temperature` | `0` | Sampling temperature (0 = deterministic) |

**Note:** Clear cache if prompts are updated:
```bash
rm -rf llm_cache_step4
rm -rf step_4_output
```

**Output per prompt** (`step_4_output/llm_{provider}_{model}_batched/{prompt_name}/`):

| File | Description |
|------|-------------|
| `predicted_boundaries_{prompt}_{params}.txt` | Predicted boundaries in same format as ground truth |
| `segmented_text_debug_{prompt}_{params}.txt` | Full text split into predicted segments |
| `batch_raw_outputs_{prompt}_{params}.csv` | Raw LLM responses per batch |
| `evaluation_output_step4_{prompt}_{params}.txt` | Evaluation report (Precision, Recall, F1, distance analysis) |

**Global output:** `step_4_output/llm_{provider}_{model}_batched/summary.csv` — run configuration and file paths for all prompts

---

### Step 5 – LLM Visualization (`step_5_visualisation_llm_grouped.py`)

Reads all evaluation files from `step_4_output/` recursively and creates grouped plots and CSV tables per combination of model, batch size, and context size.

```bash
python step_5_visualisation_llm_grouped.py
```

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALUATION_INPUT_DIR` | `step_4_output/` | Directory with step 4 evaluation files |
| `VISUALIZATION_OUTPUT_DIR` | `step_5_output/` | Output directory |
| `EVALUATION_FILE_PATTERN` | `evaluation_output_step4_*_bs*_cs*.txt` | File name pattern to match |

**Output per group** (`step_5_output/{model}_bsN_csN/`):

| File | Description |
|------|-------------|
| `*_precision_recall_f1_plot.png` | Precision, Recall, and F1 per prompt |
| `*_distance_matches_plot.png` | Boundary matches at different distance thresholds |
| `*_distance_threshold_curve.png` | Cumulative match curve across thresholds |
| `*_evaluation_results_table.csv` | All metrics per prompt as CSV |

**Global output:** `step5_all_evaluation_results_table.csv` — combined results across all groups. Metrics include: Precision, Recall, F1, Jaccard index, exact matches, within ±1/±2/±3/±5 paragraphs.

---

## Pipeline 2: Synthetic Texts Pipeline

Used for: **Robin Hood + Robinson Crusoe** mixed text variants.

### Input files

| File | Location | Description |
|------|----------|-------------|
| `robin_hood.txt` | `step_1_input/` | Robin Hood (Project Gutenberg) |
| `robinson_crusoe.txt` | `step_1_input/` | Robinson Crusoe (Project Gutenberg) |

### Step 1a – Generate Synthetic Variants (`step_1_combine_hood_crueso.py`)

Mixes Robin Hood and Robinson Crusoe paragraph by paragraph. After each paragraph there is a 5% chance of switching to the other text. Each variant uses a different random seed for reproducibility.

**How to run:**
```bash
python step_1_combine_hood_crueso.py
```

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SWITCH_PROB` | `0.05` | Probability of switching texts per paragraph (5%) |
| `N_VARIANTS` | `20` | Number of variants to generate (5 is sufficient) |
| `NUM_BATCHES` | `3` | Number of batches per variant |
| `BATCH_OVERLAP` | `4` | Overlapping **sentences** between consecutive batches |
| `MIN_LAENGE` | `150` | Minimum paragraph length in characters (shorter paragraphs filtered out) |

**Output per variant** (`step_1_output/variant_01/` … `variant_20/`):

| File | Description |
|------|-------------|
| `sentences_numbered.txt` | Sentence-level numbered text for the LLM |
| `g_grenzen.txt` | Gold standard: sentence IDs where source changes |
| `combined_text.txt` | Plain text backup of the combined variant |
| `combined_readable.txt` | Labelled version with source and paragraph info |
| `paragraph_sources.json` | Metadata per paragraph (source, original index, switches) |
| `goldstandard.json` | Full gold standard with sentence-level boundary info |
| `batches/batch_1.txt` | First batch |
| `batches/batch_2.txt` | Second batch |
| `batches/batch_3.txt` | Third batch |
| `batches/batches_metadata.json` | Batch structure (start/end indices, overlap) |

---

### Step 1b – Paragraph-level Variant (`step_1_combine_hood_crueso_paragraph_minimal.py`)

Produces only the files needed for the paragraph-level LLM pipeline, no sentence splitting, no pre-split batches. Batching is handled dynamically in Step 4.

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SWITCH_PROB` | `0.05` | Probability of switching texts per paragraph (5%) |
| `N_VARIANTS` | `20` | Number of variants to generate |
| `MIN_LAENGE` | `150` | Minimum paragraph length in characters |

**Output per variant:**

| File | Description |
|------|-------------|
| `paragraphs_numbered.txt` | Paragraph-level numbered text for the LLM |
| `g_grenzen_paragraph.txt` | Gold standard: paragraph IDs where source changes |

---

### Step 4 – LLM Segmentation

Two scripts exist depending on which Step 1 output is used:

**Original (`step_4_synth_llm.py`)** — reads `sentences_numbered.txt` + `g_grenzen.txt`:

```bash
python step_4_synth_llm.py --provider openai --model gpt-4o-mini --batch-size 40 --context-size 3 --temperature 0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--provider` | `dryrun` | LLM provider (`openai`, `ollama`, `dryrun`) |
| `--model` | `gpt-4o-mini` | Model to use |
| `--batch-size` | `40` | Number of sentences per batch |
| `--context-size` | `3` | Context sentences shown to LLM on each side of the batch |
| `--temperature` | `0` | Sampling temperature |

**Dynamic (`step_4_synth_llm_dynamic.py`)** — reads `paragraphs_numbered.txt` + `g_grenzen_paragraph.txt`:

```bash
# Dry run (no API key needed):
python step_4_synth_llm_dynamic.py

# KISSKI (meta-llama):
python step_4_synth_llm_dynamic.py --provider kisski
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-mode` | `words` | `words` (dynamic word-count) or `paragraphs` (fixed count) |
| `--min-batch-words` | `1000` | Minimum words per batch (words mode) |
| `--max-batch-words` | `1800` | Maximum words per batch (words mode) |
| `--batch-size` | `40` | Paragraphs per batch (paragraphs mode only) |
| `--dynamic-context` | `False` | Grow context window until min words reached |
| `--context-size` | `3` | Context paragraphs per side (if dynamic context disabled) |
| `--min-context-words` | `180` | Minimum words per side for dynamic context |
| `--max-context-paragraphs` | `12` | Maximum context paragraphs per side |
| `--provider` | `dryrun` | LLM provider (`openai`, `kisski`, `ollama`, `dryrun`) |
| `--model` | auto | `meta-llama-3.1-8b-instruct` (kisski), `gpt-4o-mini` (openai), `llama3.2:3b` (ollama) |
| `--max-tokens` | `128` | Maximum output tokens |
| `TEST_MODE` | `True` | If True, processes only first 2 variants (change in script to run all) |

> **Note:** `kisski` provider uses GWDG Academic Cloud (OpenAI-compatible). Set `KISSKI_API_KEY` environment variable before running.

**Output:** `step_4_output/` — LLM predicted boundaries, evaluation results, and debug files per variant and prompt

---

### Step 5 – Visualization (`step_5_synth_visualisation_llm_grouped.py`)

Reads all evaluation files from `step_4_output/` recursively and creates grouped plots and CSV tables per combination of variant, model, batch size, and context size.

```bash
python step_5_synth_visualisation_llm_grouped.py
```

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALUATION_INPUT_DIR` | `step_4_output/` | Directory with step 4 evaluation files |
| `VISUALIZATION_OUTPUT_DIR` | `step_5_output/` | Output directory |
| `EVALUATION_FILE_PATTERN` | `evaluation_output_step4_*_bs*_cs*.txt` | File name pattern to match |

**Output per group** (`step_5_output/variant_XX/{model}_bsN_csN/`):

| File | Description |
|------|-------------|
| `*_precision_recall_f1_plot.png` | Precision, Recall, and F1 per prompt |
| `*_distance_matches_plot.png` | Boundary matches at different distance thresholds |
| `*_distance_threshold_curve.png` | Cumulative match curve across thresholds |
| `*_evaluation_results_table.csv` | All metrics per prompt as CSV |

**Global output:** `step5_all_evaluation_results_table.csv` — combined results across all groups. Metrics include: Precision, Recall, F1, Jaccard index, exact matches, within ±1/±2/±3/±5 paragraphs.

---

### Evaluation (`evaluation_RobinRobinson.py`)

Compares LLM output against gold standard and computes F1-Score.

```bash
python batches/evaluation_RobinRobinson.py
```

**Parameters (set at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VARIANT` | `"variant_01"` | Which variant to evaluate (change to e.g. `"variant_05"`) |

**Output:** F1-Score per batch, saved in `batches/Results/`

---

## Prompts & Results (Dropbox)

All prompts and test results are stored in the shared **Dropbox folder (Projektarbeit B)**:

| Folder | Description |
|--------|-------------|
| `Prompts/` | Prompt .txt files tested against real texts |
| `Prompt_Tests/` | Results of manual prompt testing |
| `STEP 4/` | Step 4 LLM outputs |
| `Evaluation/` | TextTiling evaluation results for PHM and Michael Ende (`evaluation_results_table.csv`) |
| `Evaluation_Synth/` | Evaluation results for synthetic text pipeline |
| `Für Yannic/synth. Varianten/` | Step 5 outputs shared with supervisor |
| `Für Yannic/Aktuelle Skripte/` | Latest script versions (Main + Synth) |

---

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `nltk`, `openai`, `matplotlib`

---

## Contributors

Yannic, Emma, Zeynep
