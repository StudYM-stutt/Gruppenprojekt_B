from pathlib import Path

from texttiling_de import TextTilingTokenizer


INPUT_FILE = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Input"
    r"\g_standard_die_unendliche_Geschichte_ohne_Grenzen_am_Dokumentanfang.txt"
)

OUTPUT_DIR = (
    r"T:\Studium\Studium\a_Master\Digital_Humanities\2.Semester"
    r"\Projekt\Github_Projekt\ausgabe_de\Output"
)

TEXTTILING_PARAMETERS = {
    "w125_k62": {"w": 125, "k": 62},
    "w130_k65": {"w": 130, "k": 65},
    "w135_k67": {"w": 135, "k": 67},
    "w140_k70": {"w": 140, "k": 70},
    "w145_k72": {"w": 145, "k": 72},
    "w170_k85": {"w": 170, "k": 85},
    "w175_k87": {"w": 175, "k": 87},
    "w180_k90": {"w": 180, "k": 90},
    "w185_k92": {"w": 185, "k": 92},
    "w190_k95": {"w": 190, "k": 95},
}


def read_text(file_path):
    """Read the content of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def segment_text(text, w_value, k_value):
    """Segment the text using TextTiling."""
    tokenizer = TextTilingTokenizer(
        w=w_value,
        k=k_value,
    )
    return tokenizer.tokenize(text)


def write_segments(segments, file_path):
    """Write the text segments to an output file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as output_file:
        for index, segment in enumerate(segments, start=1):
            output_file.write(f"\n--- Segment {index} ---\n\n")
            output_file.write(segment)
            output_file.write("\n\n")


def create_output_file(parameter_name):
    """Create the output file path for a parameter setting."""
    return Path(OUTPUT_DIR) / f"output_step1_{parameter_name}.txt"


def run_segmentation(text, parameter_name, parameters):
    """Run TextTiling for one parameter setting."""
    w_value = parameters["w"]
    k_value = parameters["k"]
    output_file = create_output_file(parameter_name)

    segments = segment_text(text, w_value, k_value)
    write_segments(segments, output_file)

    print("Segmentation completed.")
    print(f"Parameter setting: {parameter_name}")
    print(f"w value: {w_value}")
    print(f"k value: {k_value}")
    print(f"Number of segments: {len(segments)}")
    print(f"Output saved to: {output_file}")
    print()


def main():
    """Run TextTiling for all parameter settings."""
    text = read_text(INPUT_FILE)

    for parameter_name, parameters in TEXTTILING_PARAMETERS.items():
        run_segmentation(text, parameter_name, parameters)


if __name__ == "__main__":
    main()
