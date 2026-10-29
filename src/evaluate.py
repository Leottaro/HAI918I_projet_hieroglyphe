import json
import torch

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

    # read history
    record: dict[str, any] = json.loads(history_path.read_text(encoding="utf-8"))

    history: list[dict[str, any]] = record["history"]

    if not history:
        raise ValueError("training history is empty.")

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    # collecte datas
    for item in history:
        epochs.append(item["epoch"])

        train_losses.append(item["train_loss"])
        val_losses.append(item["val_loss"])

        train_accuracies.append(item["train_accuracy"] * 100)
        val_accuracies.append(item["val_accuracy"] * 100)

    # draw figures
    figure, (loss_ax, accuracy_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    # loss figure
    loss_ax.plot(epochs, train_losses, label="Train loss")
    loss_ax.plot(epochs, val_losses, label="Validation loss")

    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    # accuracy figure
    accuracy_ax.plot(epochs, train_accuracies, label="Train accuracy")
    accuracy_ax.plot(epochs, val_accuracies, label="Validation accuracy")

    accuracy_ax.set_title("Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.set_ylim(0, 100)
    accuracy_ax.legend()
    accuracy_ax.grid(True, alpha=0.3)

    # save figures
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path: Path = (output_dir / f"{history_path.stem}_curves.png")

    figure.tight_layout()
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    print(f"Saved curves to path: {image_path}")


def collect_predictions(model: Glyphnet, dataloader: DataLoader) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)

            # get predicted score for every classes
            logits: Tensor = model(X)

            # get predicted class
            predictions: Tensor = logits.argmax(dim=1)

            # put batch data into list
            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred
                    

def report_results( y_true: list[int], y_pred: list[int], class_to_idx: dict[str, int], matrix_path: Path):

    # get label index in order
    label_indices: list[int] = sorted(class_to_idx.values())

    # remapping index to class name
    idx_to_class: dict[int, str] = {}

    for class_name, class_index in class_to_idx.items():
        idx_to_class[class_index] = class_name

    # get class name by order of index
    class_names: list[str] = []
    for class_index in label_indices:
        class_names.append(idx_to_class[class_index])

    # get report by sklearn.metrics
    report: str = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print (report)


    # confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=label_indices,
        normalize="true",
    )

    figure, ax = plt.subplots(figsize=(16, 14))

    display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        include_values=False,
    )

    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="both", labelsize=8)

    matrix_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(matrix_path, dpi=180)
    plt.close(figure)

    print(f"Saved confusion matrix to path: {matrix_path}")



def main():
    torch.manual_seed(SEED)

    # reload model and history data
    output_dir: Path = PROJECT_ROOT / "outputs"

    history_path: Path = output_dir / "training_history_epochs_30.json"

    model_path: Path = output_dir / "glyphnet_epochs_15.pt"

    matrix_path: Path = (output_dir / f"{model_path.stem}_confusion_matrix.png")

    model: Glyphnet
    trained_class_to_idx: dict[str, int]

    model, trained_class_to_idx = load_model(model_path)

    # get image paths
    image_paths: list[Path] = find_images(DATA_DIR)

    selected_classes: list[str] = list(trained_class_to_idx.keys())

    selected_image_paths: list[Path] = filter_image_paths(image_paths, selected_classes)

    # restore image split during training
    train_image_paths, val_image_paths, test_image_paths = split_dataset(
        selected_image_paths,
        SEED,
    )

    test_loader: DataLoader = create_dataloader(
        image_paths=test_image_paths,
        class_to_idx=trained_class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # draw training curves (loss and accuracy)
    plot_training_history(history_path, output_dir)

    # get predicted labels and real labels by trained model in test dataset
    y_true, y_pred = collect_predictions(model, test_loader)

    # get report
    report_results(y_true, y_pred, trained_class_to_idx, matrix_path)    


if __name__ == "__main__":
    main()