'''a test file to see if all works'''

from collections import Counter
from pathlib import Path

from config import *
from inspect_data import *
from dataset import *


def main() -> None:
    image_paths: list[Path] = find_images(DATA_DIR)

    label_counts: Counter[str] = inspect_images(image_paths)

    selected_classes: list[str] = select_top_k_class(label_counts, TOP_K_CLASSES,)

    class_to_idx: dict[str, int] = build_class_to_idx(selected_classes)

    selected_image_paths: list[Path] = filter_image_paths(image_paths, selected_classes)

    train_image_paths, val_image_paths, test_image_paths = split_dataset(
        selected_image_paths,
        SEED,
    )

    print(f"Images: {len(image_paths)}")
    print(f"Labels: {len(label_counts)}")
    print(f"Selected classes: {len(selected_classes)}")
    print(class_to_idx)
    print(f"Selected class numbers: {len(selected_image_paths)}")
    print(f"Train set numbers: {len(train_image_paths)}")
    print(f"Val set numbers: {len(val_image_paths)}")
    print(f"Test set numbers: {len(test_image_paths)}")


if __name__ == "__main__":
    main()