from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyphnet_data" / "Preprocessed"


# return a list of image(*.png) paths in this dir
def find_images(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dir not exist: {data_dir}")

    image_paths = sorted(data_dir.rglob("*.png"))

    if not image_paths:
        raise ValueError(f"no images in this dir: {data_dir}")

    return image_paths


# count all label numbers
def inspect_images(image_paths: list[Path]) -> Counter[str]:
    label_counts: Counter[str] = Counter()

    for path in image_paths:
        label:str = path.stem.rsplit("_", 1)[-1]

        label_counts[label] += 1

    return label_counts


# print label distribution
def print_report(label_counts: Counter[str]):
    total_images: int = sum(label_counts.values())
    num_labels: int = len(label_counts)

    print(f"Total images: {total_images}")
    print(f"Total labels (including UNKNOWN if present): {num_labels}")
    print(f"Total UNKNOWN images: {label_counts['UNKNOWN']}")

    print("\n Number of images per label:")

    for label, count in label_counts.most_common():
        print(f"{label}: {count}")


def main():
    image_paths: list[Path] = find_images(DATA_DIR)

    label_counts: Counter[str] = inspect_images(image_paths)

    print_report(label_counts)


if __name__ == "__main__":
    main()