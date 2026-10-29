import json
import torch

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any


from config import *
from inspect_data import *
from dataset import *

from glyphnet import *


def load_model(model_path: Path) -> tuple[Glyphnet, dict[str, int]]:
    checkpoint: dict[str, any] = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]

    model: Glyphnet = Glyphnet(num_classes=len(class_to_idx))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_to_idx


def plot_training_history(history_path: Path, output_dir: Path):

    record: dict[str, any] = json.loads(history_path.read_text(encoding="utf-8"))

    history: list[dict[str, any]] = record["history"]

    if not history:
        raise ValueError("training history is empty.")

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    for item in history:
        epochs.append(item["epoch"])

        train_losses.append(item["train_loss"])
        val_losses.append(item["val_loss"])

        train_accuracies.append(item["train_accuracy"] * 100)
        val_accuracies.append(item["val_accuracy"] * 100)

        figure, (loss_ax, accuracy_ax) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 5),
        )

        loss_ax.plot(epochs, train_losses, label="Train loss")
        loss_ax.plot(epochs, val_losses, label="Validation loss")

        loss_ax.set_title("Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()
        loss_ax.grid(True, alpha=0.3)

        accuracy_ax.plot(epochs, train_accuracies, label="Train accuracy")
        accuracy_ax.plot(epochs, val_accuracies, label="Validation accuracy")

        accuracy_ax.set_title("Accuracy")
        accuracy_ax.set_xlabel("Epoch")
        accuracy_ax.set_ylabel("Accuracy (%)")
        accuracy_ax.set_ylim(0, 100)
        accuracy_ax.legend()
        accuracy_ax.grid(True, alpha=0.3)

        output_dir.mkdir(parents=True, exist_ok=True)

        image_path: Path = (output_dir / f"{history_path.stem}_curves.png")

        figure.tight_layout()
        figure.savefig(image_path, dpi=180)
        plt.close(figure)


def collect_predictions():
    pass


def report_results():
    pass
    


def main():
    torch.manual_seed(SEED)

    image_paths: list[Path] = find_images(DATA_DIR)

    label_counts: Counter[str] = inspect_images(image_paths)

    selected_classes: list[str] = select_top_k_class(label_counts, TOP_K_CLASSES,)

    class_to_idx: dict[str, int] = build_class_to_idx(selected_classes)

    selected_image_paths: list[Path] = filter_image_paths(image_paths, selected_classes)

    train_image_paths, val_image_paths, test_image_paths = split_dataset(
        selected_image_paths,
        SEED,
    )

    test_loader: DataLoader = create_dataloader(
        image_paths=test_image_paths,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    output_dir: Path = PROJECT_ROOT / "outputs"

    history_path: Path = output_dir / "training_history_epochs_30.json"

    plot_training_history(history_path, output_dir)

    model_path: Path = output_dir / "glyphnet_epochs_15.pt"

    model: Glyphnet = load_model(model_path)




if __name__ == "__main__":
    main()