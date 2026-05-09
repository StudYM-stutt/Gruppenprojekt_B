from texttiling import TextTilingTokenizer
import numpy


INPUT_FILE = r"C:\path\to\your\input_file.txt"
OUTPUT_FILE = r"C:\path\to\your\output_file.txt"


def read_text(file_path):
    """Read the content of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def segment_text(text):
    """Segment the text using TextTiling."""
    tokenizer = TextTilingTokenizer()
    return tokenizer.tokenize(text)


def write_segments(segments, file_path):
    """Write the text segments to an output file."""
    with open(file_path, "w", encoding="utf-8") as output_file:
        for index, segment in enumerate(segments, start=1):
            output_file.write(f"\n--- Segment {index} ---\n\n")
            output_file.write(segment)
            output_file.write("\n\n")


def main():
    """Main function of the program."""
    text = read_text(INPUT_FILE)
    segments = segment_text(text)
    write_segments(segments, OUTPUT_FILE)

    print(
        "Segmentation completed. "
        f"Output saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()