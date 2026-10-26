import torch
from torch import nn, Tensor
from torch.optim import Adam

from collections import Counter
from pathlib import Path

from config import *
from inspect_data import *
from dataset import *

from glyphnet import *


print(f"Using {device} device")


def train_loop(dataloader, model, loss_fn, optimizer)-> tuple[float, float]:
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Compute prediction and loss
        pred: Tensor = model(images)
        loss: Tensor = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size: int = labels.size(0)

        predictions: Tensor = pred.argmax(dim=1)
        correct_count: int = int((predictions == labels).sum().item())

        total_loss += loss.item() * batch_size
        total_correct += correct_count
        total_samples += batch_size

    average_loss: float = total_loss / total_samples
    accuracy: float = total_correct / total_samples

    return average_loss, accuracy


def test_loop(dataloader, model, loss_fn) -> tuple[float, float]:
    model.eval()
    size = len(dataloader.dataset)
    average_loss, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            average_loss += loss_fn(pred, y).item() * y.size(0)
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_loss /= size
    accuracy /= size
    return average_loss, accuracy


def train_epochs(train_loader, val_loader, model, loss_fn, optimizer):
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_loop(train_loader, model, loss_fn, optimizer)

        print(
            f"Train loss: {train_loss:.4f}, "
            f"accuracy: {train_accuracy:.2%}"
        )

        val_loss, val_accuracy = test_loop(val_loader, model, loss_fn)

        print(
            f"Val loss: {val_loss:.4f}, "
            f"accuracy: {val_accuracy:.2%}"
        )

    print("\nFinish to train model!")


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
        shuffle=False,
        num_workers=0
    )

    test_loader: DataLoader = create_dataloader(
        image_paths=test_image_paths,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    num_classes: int = len(selected_classes)

    model = Glyphnet(num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer: Adam = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_epochs(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    print("\nStart to test model!")

    test_loss, test_accuracy = test_loop(test_loader, model, loss_fn)

    print(
        f"Test loss: {test_loss:.4f}, "
        f"accuracy: {test_accuracy:.2%}"
    )

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "glyphnet_last.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "epoch": EPOCHS,
            "image_size": IMAGE_SIZE,
        },
        output_path,
    )

    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()