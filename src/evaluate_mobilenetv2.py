"""Prototype-based evaluation and single-image prediction."""

from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from dataset_mobilenetv2 import load_mobilenet_image
from mobilenetv2 import Mobilenetv2


class AttributePrototype(TypedDict):
    class_ids: Tensor
    vectors: Tensor


class ClassAccuracy(TypedDict):
    correct: int | None
    total: int
    accuracy: float | None


class AttributeConfusionMatrix(TypedDict):
    class_names: list[str]
    counts: list[list[int]] | None


AttributePredictions = dict[str, tuple[Tensor, Tensor | None]]


class MetricCheckpoint(TypedDict):
    kind: str
    model_state_dict: dict[str, Tensor]
    attribute_names: list[str]
    attribute_value_sets: list[list[str]]
    vocabularies: NotRequired[list[list[str]]]
    image_size: tuple[int, int]
    embedding_dim: int
    prototypes: list[AttributePrototype | None]
    epoch: int


# collect the model embeddings and labels for a full dataset
@torch.no_grad()
def collect_attribute_embeddings(
    model: Mobilenetv2, loader: DataLoader, target_device: torch.device
) -> tuple[Tensor, Tensor]:
    model.eval()
    all_vectors: list[Tensor] = []
    all_labels: list[Tensor] = []
    # move each batch back to cpu for evaluation
    for images, labels in loader:
        all_vectors.append(model(images.to(target_device)).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_vectors), torch.cat(all_labels)


# build one prototype per class from the training embeddings
def build_attribute_prototypes(
    vectors: Tensor, labels: Tensor, attribute_names: tuple[str, ...]
) -> list[AttributePrototype | None]:
    prototypes: list[AttributePrototype | None] = []
    for index in range(len(attribute_names)):
        attribute_ids = labels[:, index]
        class_ids = attribute_ids[attribute_ids >= 0].unique()
        if class_ids.numel() < 2:
            # keep this attribute in the list even if it has too few classes
            prototypes.append(None)
            continue
        # get the mean training vector for each class and normalize it for cosine similarity
        centers = [
            vectors[attribute_ids == class_id, index].mean(dim=0)
            for class_id in class_ids
        ]
        prototypes.append({
            "class_ids": class_ids,
            "vectors": F.normalize(torch.stack(centers), dim=1),
        })
    return prototypes


# find the closest prototype for each vector using cosine similarity
def nearest_prototype(vectors: Tensor, prototype: AttributePrototype) -> tuple[Tensor, Tensor]:
    scores, positions = (vectors @ prototype["vectors"].T).max(dim=1)
    # positions are prototype row indices, class_ids are label IDs and can have gaps
    return prototype["class_ids"][positions], scores.clamp(-1.0, 1.0)


def collect_attribute_predictions(
    vectors: Tensor, labels: Tensor,
    prototypes: list[AttributePrototype | None], attribute_names: tuple[str, ...],
) -> AttributePredictions:
    """get the true and predicted label ids for each attribute"""
    predictions: AttributePredictions = {}
    for index, (name, prototype) in enumerate(zip(attribute_names, prototypes, strict=True)):
        # only predict for images with a known label for this attribute
        known: Tensor = labels[:, index] >= 0
        true_ids: Tensor = labels[known, index]
        predicted_ids: Tensor | None = None
        if prototype is not None:
            predicted_ids, _ = nearest_prototype(vectors[known, index], prototype)
        predictions[name] = (true_ids, predicted_ids)
    return predictions


# calculate accuracy and macro-f1 for each attribute
def summarize_attribute_predictions(
    predictions: AttributePredictions,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    accuracy: dict[str, float | None] = {}
    macro_f1: dict[str, float | None] = {}
    for name, (true_ids, predicted_ids) in predictions.items():
        # use None if there are no predictions or known labels
        if predicted_ids is None or true_ids.numel() == 0:
            accuracy[name] = None
            macro_f1[name] = None
            continue

        correct: int = int((predicted_ids == true_ids).sum().item())
        accuracy[name] = correct / true_ids.numel()
        macro_f1[name] = float(f1_score(
            true_ids, predicted_ids, average="macro", zero_division=0
        ))
    return accuracy, macro_f1


def evaluate_attributes(
    vectors: Tensor, labels: Tensor,
    prototypes: list[AttributePrototype | None], attribute_names: tuple[str, ...],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """predict each known attribute and calculate accuracy and macro-f1"""
    predictions = collect_attribute_predictions(vectors, labels, prototypes, attribute_names)
    return summarize_attribute_predictions(predictions)


# count correct predictions per class and build the confusion matrices
def summarize_class_predictions(
    predictions: AttributePredictions, attribute_value_sets: list[list[str]],
) -> tuple[dict[str, dict[str, ClassAccuracy]], dict[str, AttributeConfusionMatrix]]:
    accuracies: dict[str, dict[str, ClassAccuracy]] = {}
    matrices: dict[str, AttributeConfusionMatrix] = {}
    for (name, (true_ids, predicted_ids)), values in zip(
        predictions.items(), attribute_value_sets, strict=True,
    ):
        # keep all classes on both axes, even if some have no test examples
        counts: list[list[int]] | None = None
        if predicted_ids is not None:
            if true_ids.numel() == 0:
                counts = [[0] * len(values) for _ in values]
            else:
                counts = confusion_matrix(true_ids, predicted_ids, labels=range(len(values))).tolist()
        matrices[name] = {"class_names": values, "counts": counts}

        totals: list[int] = torch.bincount(true_ids, minlength=len(values)).tolist()
        class_results: dict[str, ClassAccuracy] = {}
        # use the matrix diagonal for the correct count of each true class
        for index, (value, total) in enumerate(zip(values, totals, strict=True)):
            correct: int | None = 0 if total == 0 else None
            if counts is not None:
                correct = counts[index][index]
            class_results[value] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if correct is not None and total else None,
            }
        accuracies[name] = class_results
    return accuracies, matrices


def plot_test_confusion_matrices(
    matrices: dict[str, AttributeConfusionMatrix], output_dir: Path,
) -> list[Path]:
    """save one normalized confusion matrix image per evaluated attribute"""
    from matplotlib.figure import Figure

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name, matrix in matrices.items():
        counts = matrix["counts"]
        if counts is None or not any(any(row) for row in counts):
            continue

        # divide each row by its true class total and keep empty rows at zero
        normalized = np.asarray(counts, dtype=float)
        row_totals = normalized.sum(axis=1, keepdims=True)
        np.divide(normalized, row_totals, out=normalized, where=row_totals != 0)

        figure = Figure(figsize=(16, 14))
        ax = figure.subplots()
        display = ConfusionMatrixDisplay(
            confusion_matrix=normalized, display_labels=matrix["class_names"],
        )
        display.plot(ax=ax, cmap="Blues", xticks_rotation=90, include_values=False)
        ax.set_title(f"Normalized confusion matrix for {name}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(axis="both", labelsize=8)

        filename = f"confusion_matrix_{'_'.join(name.split())}.png"
        image_path = output_dir / filename
        figure.tight_layout()
        figure.savefig(image_path, dpi=180)
        saved_paths.append(image_path)
    return saved_paths


# load the saved model weights, prototypes and attribute metadata
def load_metric_model(
    checkpoint_path: Path, target_device: torch.device
) -> tuple[Mobilenetv2, MetricCheckpoint]:
    saved: MetricCheckpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "ka_attribute_metric":
        raise ValueError("这里需要多属性度量学习的 checkpoint。")
    # read the label mapping from older checkpoints too
    if "attribute_value_sets" not in saved and "vocabularies" in saved:
        saved["attribute_value_sets"] = saved["vocabularies"]
    # use weights from checkpoint for backbone and heads, no need to download ImageNet weights
    model = Mobilenetv2(
        len(saved["attribute_names"]), saved["embedding_dim"], pretrained=False
    )
    model.load_state_dict(saved["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, saved


# predict the attribute values and cosine scores for one image
@torch.no_grad()
def predict_metric_image(
    model: Mobilenetv2, saved: MetricCheckpoint, image_path: Path,
    target_device: torch.device,
) -> dict[str, tuple[str, float] | None]:
    # add a batch dimension for this single image
    images = load_mobilenet_image(image_path, saved["image_size"]).unsqueeze(0)
    vectors = model(images.to(target_device)).cpu()
    result: dict[str, tuple[str, float] | None] = {}
    for index, (name, prototype) in enumerate(
        zip(saved["attribute_names"], saved["prototypes"], strict=True)
    ):
        if prototype is None:
            result[name] = None
            continue
        predicted, score = nearest_prototype(vectors[:, index], prototype)
        # turn the predicted class id back into its label
        value = saved["attribute_value_sets"][index][predicted.item()]
        result[name] = (value, score.item())
    return result


# load the configured model and print predictions for one image
def main() -> None:
    target_device = torch.device(config.device)
    model, saved = load_metric_model(config.METRIC_MODEL_PATH, target_device)
    predictions = predict_metric_image(model, saved, config.PREDICTION_IMAGE_PATH, target_device)
    for name, prediction in predictions.items():
        if prediction is None:
            print(f"{name}: insufficient training labels")
            continue
        value, cosine = prediction
        print(f"{name}: {value}  (cosine={cosine:.3f})")


if __name__ == "__main__":
    main()
