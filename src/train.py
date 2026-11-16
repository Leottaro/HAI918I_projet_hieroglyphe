import json
import pandas as pd

import torch
from torch import nn, Tensor
from torch.optim import Adam

from collections import Counter
from pathlib import Path

from config import *
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


def train_epochs(train_loader, train_eval_loader, val_loader, model, loss_fn, optimizer) -> list[dict[str, int | float]]:
    history: list[dict[str, int | float]] = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_loop(train_loader, model, loss_fn, optimizer)

        print(
            f"Train loss: {train_loss:.4f}, "
            f"accuracy: {train_accuracy:.2%}"
        )

        val_loss, val_accuracy = test_loop(val_loader, model, loss_fn)
        
        clean_train_loss, clean_train_accuracy = test_loop(train_eval_loader, model, loss_fn)
        print(f"Train (clean, eval mode) loss: {clean_train_loss:.4f}, accuracy: {clean_train_accuracy:.2%}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "clean_train_loss": clean_train_loss,
                "clean_train_accuracy": clean_train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Val loss: {val_loss:.4f}, "
            f"accuracy: {val_accuracy:.2%}"
        )

    print("\nFinish to train model!")

    return history


def main():
    torch.manual_seed(SEED)

    df: pd.DataFrame = pd.read_csv(DATA_DIR / "Tableau_imagette_MB.csv")
    df = df[~df["Deno"].str.match(R"^.* \([0-9]+\)$")] # filter to keep single sign images
    
    models: list[str] = df.columns[1:].array
    for model_name in models:       
        df2 = df[df[model_name].notna()]
    
        image_paths: list[Path] = [DATA_DIR / (deno.strip() +".png") for deno in df2["Deno"].array]
        image_size = get_min_size(image_paths)
    
        labels: list[str] = df2[model_name].array
        num_labels: int = len(set(labels))
        
        try:
            class_to_idx, train_loader, train_eval_loader, val_loader, test_loader = read_dataset(image_paths, labels, image_size)
        except:
            continue

        model = Glyphnet(num_labels).to(device)
        loss_fn = nn.CrossEntropyLoss()

        optimizer: Adam = Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        history = train_epochs(
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        training_record: dict[str, object] = {
            "config": {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "image_size": image_size,
                "num_classes": len(class_to_idx),
                "device": str(device),
            },
            "history": history,
        }

        output_dir: Path = PROJECT_ROOT / "outputs" / model_name / str(EPOCHS)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_path: Path = output_dir / "training_history.json"
        history_path.write_text(
            data=json.dumps(training_record, indent=2),
            encoding="utf-8"
        )
        print(f"\nStart to test {model_name} model!")

        test_loss, test_accuracy = test_loop(test_loader, model, loss_fn)
        print(
            f"Test loss: {test_loss:.4f}, "
            f"accuracy: {test_accuracy:.2%}"
        )

        model_path = output_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": EPOCHS,
                "image_size": image_size,
            },
            model_path,
        )
        
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()