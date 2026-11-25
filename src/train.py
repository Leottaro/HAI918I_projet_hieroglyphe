import argparse
import json
import shutil
import pandas as pd

import torch
from torch import nn, Tensor
from torch.optim import Adam

from collections import Counter
from pathlib import Path

from config import *
from dataset import *

from evaluate import *
from glyphnet import *

print(f"Using {DEVICE} device")


def train_loop(dataloader, model, loss_fn, optimizer)-> tuple[float, float]:
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    model.train()
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            average_loss += loss_fn(pred, y).item() * y.size(0)
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_loss /= size
    accuracy /= size
    return average_loss, accuracy


def train_epochs(train_loader, clean_train_loader, val_loader, model, loss_fn, optimizer, verbose: bool) -> list[dict[str, int | float]]:
    history: list[dict[str, int | float]] = []
    
    for epoch in range(CONFIG.epochs):
        print(f"Epoch {epoch + 1}/{CONFIG.epochs}")

        train_loss, train_accuracy = train_loop(train_loader, model, loss_fn, optimizer)

        if verbose:
            print(f"\tTrain loss: {train_loss:.4f}, \taccuracy: {train_accuracy:.2%}")

        val_loss, val_accuracy = test_loop(val_loader, model, loss_fn)
        
        clean_train_loss, clean_train_accuracy = test_loop(clean_train_loader, model, loss_fn)
        if verbose:
            print(f"\tTrain (clean, eval mode) loss: {clean_train_loss:.4f}, accuracy: {clean_train_accuracy:.2%}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "clean_train_loss": clean_train_loss,
            "clean_train_accuracy": clean_train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        if verbose:
            print(f"\tVal loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}")
        
        if val_accuracy == 1.0 and val_loss == 0.0:
            print(("\t" if verbose else "") + f"Perfect model, skipping other epoch...")
            break

    return history


def main(verbose: bool = True):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="non verbose print")
    args = parser.parse_args()
    if args.quiet:
        verbose = False
    
    torch.manual_seed(CONFIG.seed)

    df: pd.DataFrame = pd.read_csv(DATA_DIR / "Tableau_imagette_MB.csv")
    df = df[~df["Deno"].str.match(R"^.* \([0-9]+\)$")] # filter to keep single sign images
    
    models: list[str] = df.columns[1:].array
    for model_name in models: 
        if model_name not in df.columns[1:].array:
            print(f"\"{model_name}\" should be in {df.columns[1:].array}\nskipping...")
            continue
        print(("" if verbose else "\n\n\n\n") + f"Training {model_name} model...")
        
        df2 = df[df[model_name].groupby(df[model_name]).transform('size') >= 6]
    
        image_paths: list[Path] = [DATA_DIR / (deno.strip() +".png") for deno in df2["Deno"].array]
        labels: list[str] = df2[model_name].array
        num_labels: int = len(set(labels))
        
        try:
            class_to_idx, train_loader, clean_train_loader, val_loader, test_loader = read_dataset(image_paths, labels)
        except ValueError as e:
            print(f"couldn't read dataset: {e}")
            continue

        model = Glyphnet(num_labels).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()

        optimizer: Adam = Adam(
            model.parameters(),
            lr=CONFIG.learning_rate,
        )

        history = train_epochs(
            train_loader=train_loader,
            clean_train_loader=clean_train_loader,
            val_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            verbose=verbose,
        )

        print(("\n" if verbose else "\n\n") + f"Testing {model_name} model...")
        test_loss, test_accuracy = test_loop(test_loader, model, loss_fn)
        if verbose:
            print(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.2%}")
        
        output_dir: Path = OUTPUT_DIR / CONFIG.getDirName() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": CONFIG.epochs,
                "image_size": CONFIG.image_size,
            },
            model_path,
        )
        
        print(f"\nGenerating {model_name} figures...")
        
        y_true, y_pred = collect_predictions(model, test_loader)
        matrix_path: Path = output_dir / f"confusion_matrix_{test_accuracy}acc_{test_loss}loss.png"
        report: dict[str, Any] = report_results(y_true, y_pred, class_to_idx, matrix_path, model_name, verbose)
        
        history_path: Path = output_dir / "training_history.json"
        training_record: dict[str, object] = {
            "config": CONFIG,
            "report": report,
            "history": history,
        }
        history_path.write_text(
            data=json.dumps(training_record, indent=2),
            encoding="utf-8"
        )
        plot_training_history(history_path, output_dir, model_name, verbose)
        
        best_history_path: Path = BEST_CONFIG_DIR / model_name / "training_history.json"
        best_report: float | None = None
        if best_history_path.exists():
            try:
                best_report = float(json.loads(best_history_path.read_text(encoding="utf-8"))["report"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if best_report is None or report["macro avg"]["f1-score"] > best_report["macro avg"]["f1-score"]:
            best_model_dir: Path = BEST_CONFIG_DIR / model_name
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            shutil.copytree(output_dir, best_model_dir)
        
        if verbose:
            print(f"saved everything in: {output_dir}")

if __name__ == "__main__":
    main()