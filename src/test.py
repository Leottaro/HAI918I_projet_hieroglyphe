'''a test file to see if all works'''

from collections import Counter
from pathlib import Path

from config import *
from inspect_data import *
from dataset import *

from glyphnet import *


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

    train_loader: DataLoader = create_dataloader(
        image_paths=train_image_paths,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader: DataLoader = create_dataloader(
        image_paths=val_image_paths,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader: DataLoader = create_dataloader(
        image_paths=test_image_paths,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    print(f"Images: {len(image_paths)}")
    print(f"Labels: {len(label_counts)}")
    print(f"\nSelected classes: {len(selected_classes)}")
    print(class_to_idx)
    print(f"Selected class numbers: {len(selected_image_paths)}")
    print(f"\nTrain set numbers: {len(train_image_paths)}")
    print(f"Val set numbers: {len(val_image_paths)}")
    print(f"Test set numbers: {len(test_image_paths)}")

    for images, labels in train_loader:
        print(f"\nImages shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        break

    num_classes: int = len(selected_classes)

    model = Glyphnet(num_classes).to(device)

    model.eval()
        
    x: Tensor = torch.randn(1, 1, 75, 50, device=device,)
    
    summary(
        model,
        input_data=x,
        col_names=("input_size", "output_size", "num_params"),
        depth=3,
        mode="eval",
    )


if __name__ == "__main__":
    main()