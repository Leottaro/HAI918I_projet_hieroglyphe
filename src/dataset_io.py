import json
from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp"}
)


""" find all images in this file path"""
def find_image_paths(directory: Path) -> list[Path]:
    image_paths: list[Path] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            image_paths.append(path)

    image_paths.sort(key=lambda path: path.name.casefold())
    return image_paths


""" get a output path, when repeate: name (1).png、name (2).png """
def get_available_output_path(directory: Path, filename: str) -> Path:
    filename = filename.strip()
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("Please enter a valid filename without a directory")

    base_path: Path = (directory / filename).with_suffix(".png")
    output_path: Path = base_path
    number: int = 1
    while output_path.exists():
        # always use the base name + number
        output_path = base_path.with_name(
            f"{base_path.stem}{number:03d}{base_path.suffix}"
        )
        number += 1
    return output_path


""" save processed image and return its path, dont overwrite old files"""
def save_processed_image(
    image: Image.Image,
    directory: Path,
    filename: str,
) -> Path:
    output_path: Path = get_available_output_path(directory, filename)
    with output_path.open("xb") as file:
        image.save(file, format="PNG")
    return output_path


""" add a record into metadata"""
def append_sample_metadata(
    directory: Path,
    output_path: Path,
    source_path: Path,
    label: str | None,
    image_size: tuple[int, int],
) -> None:
    record: dict[str, object] = {
        "output_file": output_path.name,
        "source_file": source_path.name,
        "label": label,
        "image_size": list(image_size),
    }
    metadata_path: Path = directory / "metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
