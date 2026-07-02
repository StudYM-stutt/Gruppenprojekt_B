"""
Step 4: LLM-based narrative segmentation.

Goal:
- Read the numbered project text.
- Run one or more prompt files against the text in paragraph batches.
- Produce predicted_boundaries.txt in the same simple TXT structure as g_grenzen.txt:

  first line  = total number of predicted boundaries
  next lines  = paragraph IDs where a new segment begins

This script writes the Step 4 prediction files and additionally creates one evaluation_output_step4.txt per prompt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # only needed for Ollama
    requests = None


BASE_DIR = Path(__file__).resolve().parent

STEP1_OUTPUT_DIR = BASE_DIR / "step_1_output"
OUTPUT_DIR = BASE_DIR / "step_4_output"
CACHE_DIR = BASE_DIR / "llm_cache_step4"
PROMPT_DIR = BASE_DIR / "step_4_prompt_input"

# Used only to keep cached LLM responses separated by synthetic variant.
CURRENT_VARIANT_NAME = "single_dataset"


USER_PROMPT_TEMPLATE = """
Du bekommst einen Ausschnitt aus einem narrativen Text.

Die Textabschnitte sind nummeriert. Entscheide, bei welchen Abschnittsnummern
innerhalb des zu prüfenden Batchs ein neues narratives Segment beginnt.

Definition:
Eine Segmentgrenze liegt bei der Abschnittsnummer, an der ein neues narratives
Segment beginnt. Die Abschnittsnummer ist also der erste Absatz des neuen Segments.

Regeln:
- Gib nur Abschnittsnummern aus dem Bereich "Zu prüfender Batch" zurück.
- Die Kontextabschnitte dienen nur zur Orientierung und dürfen nicht ausgegeben werden.
- Abschnitt [0] bzw. der erste Abschnitt des gesamten Textes wird nie als Segmentgrenze ausgegeben.
- Gib den letzten Absatz des Batchs nicht nur deshalb aus, weil der Batch endet.
- Du kennst keinen Goldstandard und sollst selbst eine Vorhersage treffen.

Ausgabeformat:
- Antworte ausschließlich mit Zahlen.
- Erste Zeile: Anzahl der vorhergesagten Segmentgrenzen in diesem Batch.
- Danach: genau eine Abschnittsnummer pro Zeile.
- Wenn der Batch keine neue Segmentgrenze enthält, antworte ausschließlich mit:
0

Beispiel mit drei Grenzen:
3
8
14
19

Vorheriger Kontext:
{left_context}

Zu prüfender Batch:
{batch_paragraphs}

Nachfolgender Kontext:
{right_context}
""".strip()


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    text: str


@dataclass(frozen=True)
class PromptConfig:
    name: str
    path: Path
    system_prompt: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Batch:
    batch_index: int
    start_index: int
    end_index: int
    paragraphs: list[Paragraph]


@dataclass(frozen=True)
class BatchDecision:
    prompt_name: str
    batch_index: int
    batch_start_id: int
    batch_end_id: int
    boundaries: set[int]
    raw_response: str
    parser_warning: str


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def load_prompts(prompt_dir: Path) -> list[PromptConfig]:
    if not prompt_dir.exists():
        raise FileNotFoundError(
            f"Prompt directory not found: {prompt_dir}\n"
            "Create a folder named 'prompts' next to this script and put your prompt .txt files there."
        )

    prompt_files = sorted(prompt_dir.glob("*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No .txt prompt files found in: {prompt_dir}")

    prompts: list[PromptConfig] = []
    for prompt_file in prompt_files:
        prompt_text = read_text(prompt_file).strip()
        if prompt_text:
            prompts.append(
                PromptConfig(
                    name=slugify(prompt_file.stem),
                    path=prompt_file,
                    system_prompt=prompt_text,
                )
            )
        else:
            print(f"Skipping empty prompt file: {prompt_file}")

    if not prompts:
        raise ValueError("All prompt files are empty.")
    return prompts


def parse_numbered_paragraphs(text: str) -> list[Paragraph]:
    pattern = re.compile(r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)")
    matches = list(pattern.finditer(text))

    if matches:
        paragraphs = [
            Paragraph(paragraph_id=int(match.group(1)), text=match.group(2).strip())
            for match in matches
        ]
    else:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        paragraphs = [Paragraph(paragraph_id=index, text=block) for index, block in enumerate(blocks)]

    if not paragraphs:
        raise ValueError("No paragraphs found in input text.")

    paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    duplicate_ids = sorted({pid for pid in paragraph_ids if paragraph_ids.count(pid) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate paragraph IDs found: {duplicate_ids[:20]}")

    if paragraph_ids != sorted(paragraph_ids):
        raise ValueError("Paragraph IDs are not sorted in ascending order.")

    return paragraphs


def format_context(paragraphs: Iterable[Paragraph]) -> str:
    lines = [f"[{paragraph.paragraph_id}] {paragraph.text}" for paragraph in paragraphs]
    return "\n\n".join(lines) if lines else "(kein Kontext)"


def create_batches(paragraphs: list[Paragraph], batch_size: int) -> list[Batch]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    batches: list[Batch] = []
    for batch_index, start_index in enumerate(range(0, len(paragraphs), batch_size), start=1):
        end_index = min(start_index + batch_size, len(paragraphs))
        batches.append(
            Batch(
                batch_index=batch_index,
                start_index=start_index,
                end_index=end_index,
                paragraphs=paragraphs[start_index:end_index],
            )
        )
    return batches


def build_batch_prompt(paragraphs: list[Paragraph], batch: Batch, context_size: int) -> str:
    if context_size < 0:
        raise ValueError("context_size must be 0 or greater.")

    left_start = max(0, batch.start_index - context_size)
    right_end = min(len(paragraphs), batch.end_index + context_size)

    return USER_PROMPT_TEMPLATE.format(
        left_context=format_context(paragraphs[left_start:batch.start_index]),
        batch_paragraphs=format_context(batch.paragraphs),
        right_context=format_context(paragraphs[batch.end_index:right_end]),
    )


def cache_path(provider: str, model: str, prompt_config: PromptConfig, batch: Batch) -> Path:
    start_id = batch.paragraphs[0].paragraph_id
    end_id = batch.paragraphs[-1].paragraph_id
    return (
        CACHE_DIR
        / CURRENT_VARIANT_NAME
        / f"{slugify(provider)}_{slugify(model)}"
        / f"{prompt_config.name}_{prompt_config.content_hash}"
        / f"batch_{batch.batch_index:04d}_{start_id}_{end_id}.txt"
    )


def load_cached_batch(provider: str, model: str, prompt_config: PromptConfig, batch: Batch) -> str | None:
    path = cache_path(provider, model, prompt_config, batch)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def save_cached_batch(provider: str, model: str, prompt_config: PromptConfig, batch: Batch, raw_response: str) -> None:
    path = cache_path(provider, model, prompt_config, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_response, encoding="utf-8")


def call_openai(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def call_ollama(model: str, system_prompt: str, user_prompt: str, temperature: float, ollama_url: str) -> str:
    if requests is None:
        raise ImportError("Install requests first: pip install requests")

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str, temperature: float, ollama_url: str) -> str:
    if provider == "openai":
        return call_openai(model, system_prompt, user_prompt, temperature)
    if provider == "ollama":
        return call_ollama(model, system_prompt, user_prompt, temperature, ollama_url)
    if provider == "dryrun":
        return "0"
    raise ValueError(f"Unsupported provider: {provider}")


def parse_boundary_response(raw_response: str, allowed_ids: set[int]) -> tuple[set[int], str]:
    """
    Expected LLM format per batch:
      0
    or
      N
      boundary_1
      boundary_2
      ...

    Returns accepted boundaries and a warning string for debug output.
    """
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:txt|text)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    if not cleaned:
        return set(), "empty_response"

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not all(re.fullmatch(r"\d+", line) for line in lines):
        return set(), "non_numeric_response_ignored"

    numbers = [int(line) for line in lines]

    if numbers == [0]:
        return set(), ""

    declared_count = numbers[0]
    candidate_boundaries = numbers[1:]

    warning = ""
    if declared_count != len(candidate_boundaries):
        warning = f"count_mismatch_declared_{declared_count}_parsed_{len(candidate_boundaries)}"

    accepted = {number for number in candidate_boundaries if number in allowed_ids and number != 0}

    rejected = [number for number in candidate_boundaries if number not in allowed_ids or number == 0]
    if rejected:
        warning = (warning + " | " if warning else "") + f"out_of_batch_or_zero_ignored_{rejected}"

    return accepted, warning


def predict_boundaries_batched(
    paragraphs: list[Paragraph],
    prompt_config: PromptConfig,
    provider: str,
    model: str,
    batch_size: int,
    context_size: int,
    temperature: float,
    ollama_url: str,
    sleep_seconds: float,
    use_cache: bool,
) -> tuple[set[int], list[BatchDecision]]:
    batches = create_batches(paragraphs, batch_size)
    all_boundaries: set[int] = set()
    batch_decisions: list[BatchDecision] = []
    first_paragraph_id = paragraphs[0].paragraph_id

    for batch in batches:
        start_id = batch.paragraphs[0].paragraph_id
        end_id = batch.paragraphs[-1].paragraph_id
        print(f"Checking batch {batch.batch_index}/{len(batches)} ([{start_id}]–[{end_id}]) ...")

        raw_response = load_cached_batch(provider, model, prompt_config, batch) if use_cache else None
        if raw_response is None:
            user_prompt = build_batch_prompt(paragraphs, batch, context_size)
            raw_response = call_llm(
                provider=provider,
                model=model,
                system_prompt=prompt_config.system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                ollama_url=ollama_url,
            )
            if use_cache:
                save_cached_batch(provider, model, prompt_config, batch, raw_response)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        allowed_ids = {paragraph.paragraph_id for paragraph in batch.paragraphs}
        allowed_ids.discard(first_paragraph_id)
        boundaries, parser_warning = parse_boundary_response(raw_response, allowed_ids)

        all_boundaries.update(boundaries)
        batch_decisions.append(
            BatchDecision(
                prompt_name=prompt_config.name,
                batch_index=batch.batch_index,
                batch_start_id=start_id,
                batch_end_id=end_id,
                boundaries=boundaries,
                raw_response=raw_response,
                parser_warning=parser_warning,
            )
        )

        if parser_warning:
            print(f"  Parser warning: {parser_warning}")

    return all_boundaries, batch_decisions


def create_segments(paragraphs: list[Paragraph], boundaries: set[int]) -> list[list[Paragraph]]:
    segments: list[list[Paragraph]] = []
    current_segment: list[Paragraph] = []

    for paragraph in paragraphs:
        if paragraph.paragraph_id in boundaries and current_segment:
            segments.append(current_segment)
            current_segment = []
        current_segment.append(paragraph)

    if current_segment:
        segments.append(current_segment)
    return segments


def write_boundary_list(boundaries: set[int], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_boundaries = sorted(boundary for boundary in boundaries if boundary != 0)

    with output_file.open("w", encoding="utf-8") as file:
        file.write(f"{len(sorted_boundaries)}\n")
        for boundary in sorted_boundaries:
            file.write(f"{boundary}\n")


def load_ground_truth(path: Path) -> tuple[int, set[int]]:
    """
    Read the gold standard boundary file.

    Expected format:
    first line: total number of boundaries
    following lines: paragraph IDs where a new segment begins
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Ground truth file is empty: {path}")

    total_boundaries = int(lines[0])
    boundaries = {int(line) for line in lines[1:]}

    return total_boundaries, boundaries


def evaluate_boundaries(
    predicted_boundaries: set[int],
    ground_truth_boundaries: set[int],
) -> tuple[set[int], int, int, int, float, float, float]:
    """Calculate precision, recall, and F1 score."""
    true_positive_boundaries = predicted_boundaries & ground_truth_boundaries

    true_positives = len(true_positive_boundaries)
    false_positives = len(predicted_boundaries - ground_truth_boundaries)
    false_negatives = len(ground_truth_boundaries - predicted_boundaries)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = true_positives / precision_denominator if precision_denominator > 0 else 0
    recall = true_positives / recall_denominator if recall_denominator > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return (
        true_positive_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    )


def calculate_boundary_distances(
    predicted_boundaries: set[int],
    ground_truth_boundaries: set[int],
) -> list[int]:
    """Calculate distances from each gold boundary to the nearest predicted boundary."""
    if not predicted_boundaries:
        return []

    distances = []
    for ground_truth_boundary in sorted(ground_truth_boundaries):
        nearest_prediction = min(
            predicted_boundaries,
            key=lambda boundary: abs(boundary - ground_truth_boundary),
        )
        distances.append(abs(nearest_prediction - ground_truth_boundary))

    return distances


def calculate_distance_matches(distances: list[int]) -> dict[str, int]:
    """Calculate matches within selected paragraph distance thresholds."""
    return {
        "exact_matches": sum(distance == 0 for distance in distances),
        "matches_within_1": sum(distance <= 1 for distance in distances),
        "matches_within_2": sum(distance <= 2 for distance in distances),
        "matches_within_3": sum(distance <= 3 for distance in distances),
        "matches_within_5": sum(distance <= 5 for distance in distances),
    }


def write_evaluation_report(
    predicted_boundaries: set[int],
    ground_truth_file: Path,
    output_file: Path,
    prompt_name: str,
    boundary_output_file: Path,
) -> None:
    """Write an evaluation report in the same style as evaluation_output_step1_w20_k10.txt."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_ground_truth_boundaries, ground_truth_boundaries = load_ground_truth(ground_truth_file)
    predicted_boundaries = {boundary for boundary in predicted_boundaries if boundary != 0}

    (
        true_positive_boundaries,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    ) = evaluate_boundaries(predicted_boundaries, ground_truth_boundaries)

    distances = calculate_boundary_distances(predicted_boundaries, ground_truth_boundaries)
    distance_matches = calculate_distance_matches(distances)

    with output_file.open("w", encoding="utf-8") as file:
        file.write("=== Evaluation Step 4 LLM ===\n\n")
        file.write(f"Step 4 output: {boundary_output_file}\n")
        file.write(f"Prompt: {prompt_name}\n")
        file.write(f"Ground truth: {ground_truth_file}\n\n")

        file.write(f"Ground truth boundaries according to file: {total_ground_truth_boundaries}\n")
        file.write(f"Read ground truth boundaries: {len(ground_truth_boundaries)}\n")
        file.write(f"Predicted boundaries: {len(predicted_boundaries)}\n\n")

        file.write(f"TP: {true_positives}\n")
        file.write(f"FP: {false_positives}\n")
        file.write(f"FN: {false_negatives}\n\n")

        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"F1 score:  {f1_score:.4f}\n\n")

        file.write("Distance analysis\n")
        file.write("-----------------\n")
        file.write(f"Exact matches: {distance_matches['exact_matches']}\n")
        file.write(f"Matches within +/-1 paragraph: {distance_matches['matches_within_1']}\n")
        file.write(f"Matches within +/-2 paragraphs: {distance_matches['matches_within_2']}\n")
        file.write(f"Matches within +/-3 paragraphs: {distance_matches['matches_within_3']}\n")
        file.write(f"Matches within +/-5 paragraphs: {distance_matches['matches_within_5']}\n\n")

        file.write("True positive boundaries:\n")
        for boundary in sorted(true_positive_boundaries):
            file.write(f"{boundary}\n")


def write_segments(segments: list[list[Paragraph]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            file.write(f"\n--- Segment {index} ---\n\n")
            for paragraph in segment:
                file.write(f"[{paragraph.paragraph_id}] {paragraph.text}\n\n")


def write_batch_outputs(batch_decisions: list[BatchDecision], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "prompt_name",
                "batch_index",
                "batch_start_id",
                "batch_end_id",
                "parsed_boundaries",
                "parser_warning",
                "raw_response",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for decision in batch_decisions:
            writer.writerow(
                {
                    "prompt_name": decision.prompt_name,
                    "batch_index": decision.batch_index,
                    "batch_start_id": decision.batch_start_id,
                    "batch_end_id": decision.batch_end_id,
                    "parsed_boundaries": ",".join(str(value) for value in sorted(decision.boundaries)),
                    "parser_warning": decision.parser_warning,
                    "raw_response": decision.raw_response,
                }
            )


def write_summary(summary_rows: list[dict[str, str | int | float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_name",
        "prompt_file",
        "provider",
        "model",
        "batch_size",
        "context_size",
        "predicted_boundaries",
        "boundary_output_file",
        "segment_output_file",
        "batch_output_file",
        "evaluation_output_file",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 4 LLM segmentation for all synthetic variants and "
            "write one evaluation report per variant and prompt."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=STEP1_OUTPUT_DIR,
        help="Directory containing variant_XX folders from the synthetic text generation step.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    parser.add_argument("--provider", choices=["openai", "ollama", "dryrun"], default="dryrun")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between API calls.")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def get_variant_dirs(input_dir: Path) -> list[Path]:
    """Return all synthetic variant folders that contain the required files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    variant_dirs = sorted(
        path for path in input_dir.glob("variant_*")
        if path.is_dir()
    )

    if not variant_dirs:
        raise FileNotFoundError(
            f"No variant_* folders found in: {input_dir}"
        )

    valid_variant_dirs = []
    for variant_dir in variant_dirs:
        input_file = variant_dir / "sentences_numbered.txt"
        ground_truth_file = variant_dir / "g_grenzen.txt"

        if not input_file.exists():
            print(f"Skipping {variant_dir.name}: missing {input_file.name}")
            continue

        if not ground_truth_file.exists():
            print(f"Skipping {variant_dir.name}: missing {ground_truth_file.name}")
            continue

        valid_variant_dirs.append(variant_dir)

    if not valid_variant_dirs:
        raise FileNotFoundError(
            f"No usable variant folders found in: {input_dir}. "
            "Each variant needs sentences_numbered.txt and g_grenzen.txt."
        )

    return valid_variant_dirs


def append_global_summary(
    summary_rows: list[dict[str, str | int | float]],
    output_file: Path,
) -> None:
    """Write a cross-variant summary file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "prompt_name",
        "prompt_file",
        "provider",
        "model",
        "batch_size",
        "context_size",
        "predicted_boundaries",
        "boundary_output_file",
        "segment_output_file",
        "batch_output_file",
        "evaluation_output_file",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(summary_rows)

def main() -> None:
    global CURRENT_VARIANT_NAME

    args = parse_args()

    prompt_configs = load_prompts(args.prompt_dir)
    variant_dirs = get_variant_dirs(args.input_dir)

    print(f"Loaded prompts: {len(prompt_configs)}")
    print(f"Loaded variants: {len(variant_dirs)}")
    print(f"Provider/model: {args.provider}/{args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context size: {args.context_size}")

    global_summary_rows: list[dict[str, str | int | float]] = []
    parameter_suffix = f"bs{args.batch_size}_cs{args.context_size}"

    for variant_dir in variant_dirs:
        CURRENT_VARIANT_NAME = variant_dir.name

        input_file = variant_dir / "sentences_numbered.txt"
        ground_truth_file = variant_dir / "g_grenzen.txt"

        print(f"\n=== Running variant: {variant_dir.name} ===")
        print(f"Input: {input_file}")
        print(f"Ground truth: {ground_truth_file}")

        paragraphs = parse_numbered_paragraphs(read_text(input_file))

        if len(paragraphs) < 2:
            raise ValueError(
                f"The input text needs at least two paragraphs: {input_file}"
            )

        print(f"Loaded paragraphs: {len(paragraphs)}")

        run_output_dir = (
            args.output_dir
            / variant_dir.name
            / f"llm_{args.provider}_{slugify(args.model)}_batched"
        )
        variant_summary_rows: list[dict[str, str | int | float]] = []

        for prompt_config in prompt_configs:
            print(f"\n=== Running prompt: {prompt_config.name} ===")
            print(f"Prompt file: {prompt_config.path}")

            predicted_boundaries, batch_decisions = predict_boundaries_batched(
                paragraphs=paragraphs,
                prompt_config=prompt_config,
                provider=args.provider,
                model=args.model,
                batch_size=args.batch_size,
                context_size=args.context_size,
                temperature=args.temperature,
                ollama_url=args.ollama_url,
                sleep_seconds=args.sleep,
                use_cache=not args.no_cache,
            )

            prompt_output_dir = run_output_dir / prompt_config.name

            boundary_output_file = (
                prompt_output_dir
                / f"predicted_boundaries_{prompt_config.name}_{parameter_suffix}.txt"
            )

            segment_output_file = (
                prompt_output_dir
                / f"segmented_text_debug_{prompt_config.name}_{parameter_suffix}.txt"
            )

            batch_output_file = (
                prompt_output_dir
                / f"batch_raw_outputs_{prompt_config.name}_{parameter_suffix}.csv"
            )

            evaluation_output_file = (
                prompt_output_dir
                / f"evaluation_output_step4_{prompt_config.name}_{parameter_suffix}.txt"
            )

            write_boundary_list(predicted_boundaries, boundary_output_file)
            write_segments(create_segments(paragraphs, predicted_boundaries), segment_output_file)
            write_batch_outputs(batch_decisions, batch_output_file)
            write_evaluation_report(
                predicted_boundaries=predicted_boundaries,
                ground_truth_file=ground_truth_file,
                output_file=evaluation_output_file,
                prompt_name=prompt_config.name,
                boundary_output_file=boundary_output_file,
            )

            print("\nBatched LLM segmentation completed.")
            print(f"Variant: {variant_dir.name}")
            print(f"Prompt: {prompt_config.name}")
            print(f"Number of predicted boundaries: {len(predicted_boundaries)}")
            print(f"Boundary list saved to: {boundary_output_file}")
            print(f"Evaluation report saved to: {evaluation_output_file}")

            summary_row = {
                "variant": variant_dir.name,
                "prompt_name": prompt_config.name,
                "prompt_file": prompt_config.path.name,
                "provider": args.provider,
                "model": args.model,
                "batch_size": args.batch_size,
                "context_size": args.context_size,
                "predicted_boundaries": len(predicted_boundaries),
                "boundary_output_file": str(boundary_output_file.relative_to(args.output_dir)),
                "segment_output_file": str(segment_output_file.relative_to(args.output_dir)),
                "batch_output_file": str(batch_output_file.relative_to(args.output_dir)),
                "evaluation_output_file": str(evaluation_output_file.relative_to(args.output_dir)),
            }
            variant_summary_rows.append(summary_row)
            global_summary_rows.append(summary_row)

        summary_file = run_output_dir / "summary.csv"
        write_summary(
            [
                {key: value for key, value in row.items() if key != "variant"}
                for row in variant_summary_rows
            ],
            summary_file,
        )
        print(f"\nVariant completed. Summary saved to: {summary_file}")

    global_summary_file = (
        args.output_dir
        / f"llm_{args.provider}_{slugify(args.model)}_batched_summary_all_variants.csv"
    )
    append_global_summary(global_summary_rows, global_summary_file)
    print(f"\nAll variant runs completed. Global summary saved to: {global_summary_file}")


if __name__ == "__main__":
    main()
