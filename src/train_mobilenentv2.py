"""Train and evaluate the MobileNetV2 attribute embedding model."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import config
from dataset_mobilenetv2 import (
    AttributeDataset,
    build_attribute_value_sets,
    read_attribute_samples,
    split_attribute_samples,
)
from evaluate_mobilenetv2 import (
    AttributeConfusionMatrix,
    ClassAccuracy,
    MetricCheckpoint,
    build_attribute_prototypes,
    collect_attribute_embeddings,
    collect_attribute_predictions,
    evaluate_attributes,
    load_metric_model,
    plot_test_confusion_matrices,
    summarize_attribute_predictions,
    summarize_class_predictions,
)
from mobilenetv2 import Mobilenetv2, attribute_contrastive_loss


# train for one epoch and return the total and per-attribute losses
def train_metric_epoch(
    model: Mobilenetv2, loader: DataLoader, optimizer: Optimizer,
    target_device: torch.device,
    *, trainable_attribute_indices: list[int] | None = None,
) -> tuple[float, dict[int, float]]:
    model.train()
    batch_losses: list[float] = []
    attribute_batch_losses: defaultdict[int, list[float]] = defaultdict(list)

    for images, attribute_ids in loader:
        optimizer.zero_grad()
        vectors = model(images.to(target_device))
        loss, losses_by_index = attribute_contrastive_loss(
            vectors, attribute_ids.to(target_device), config.METRIC_MARGIN,
            attribute_indices=trainable_attribute_indices,
        )
        # skip batches with no valid image pairs
        if loss is None:
            continue

        # backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()

        # detach the batch losses before saving their values for logging
        loss_values: list[float] = torch.stack([loss, *losses_by_index.values()]).detach().tolist()
        batch_losses.append(loss_values[0])
        for index, value in zip(losses_by_index, loss_values[1:], strict=True):
            attribute_batch_losses[index].append(value)
    if not batch_losses:
        raise ValueError("No available image pairs")

    # average each attribute loss over the batches where it was calculated
    mean_attribute_losses = {
        index: sum(values) / len(values) for index, values in attribute_batch_losses.items()
    }
    return sum(batch_losses) / len(batch_losses), mean_attribute_losses


def format_metric(value: float | None, spec: str) -> str:
    """show a dash for missing values and format the other scores"""
    return "—" if value is None else format(value, spec)


# print class scores and confusion matrices for the test set
def print_test_results(
    epoch: int,
    accuracy: dict[str, float | None],
    macro_f1: dict[str, float | None],
    class_accuracy: dict[str, dict[str, ClassAccuracy]],
) -> None:
    print(f"\nTest set | Best epoch {epoch:02d}")
    for name, class_results in class_accuracy.items():
        print(
            f"\n{name} | Accuracy {format_metric(accuracy[name], '.1%')} | "
            f"Macro-F1 {format_metric(macro_f1[name], '.3f')}"
        )
        class_width = max(map(len, ("Class", *class_results)))
        print(f"{'ID':>3}  {'Class':<{class_width}}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        for index, (value, metrics) in enumerate(class_results.items()):
            correct = "—" if metrics["correct"] is None else str(metrics["correct"])
            class_score = format_metric(metrics["accuracy"], ".1%")
            print(f"{index:>3}  {value:<{class_width}}  {correct:>7}  {metrics['total']:>5}  {class_score:>8}")

# train the model, save the best checkpoint and evaluate it on the test set
def train_metric_model(
    csv_path: Path, checkpoint_path: Path, epochs: int, target_device: torch.device,
    *, pretrained: bool = True, outputs_dir: Path | None = None,
) -> None:
    if epochs < 1:
        raise ValueError("EPOCHS must be bigger than 0。")

    torch.manual_seed(config.SEED)
    attribute_names = config.ATTRIBUTES
    image_size = config.IMAGE_SIZE
    batch_size = config.BATCH_SIZE

    samples = read_attribute_samples(csv_path, attribute_names)
    train_samples, val_samples, test_samples = split_attribute_samples(samples, seed=config.SEED)
    # share the same label ids across splits and check which attributes we can train
    attribute_value_sets = build_attribute_value_sets(samples, len(attribute_names))
    train_value_sets = build_attribute_value_sets(train_samples, len(attribute_names))
    trainable_attribute_indices = [
        index for index, values in enumerate(train_value_sets) if len(values) >= 2
    ]
    if not trainable_attribute_indices:
        raise ValueError("No attribute has at least two labelled values in the training set.")
    unsupported_attributes = [
        name for name, values in zip(attribute_names, train_value_sets, strict=True)
        if len(values) < 2
    ]

    train_set = AttributeDataset(train_samples, attribute_value_sets, image_size)
    val_set = AttributeDataset(val_samples, attribute_value_sets, image_size)
    test_set = AttributeDataset(test_samples, attribute_value_sets, image_size)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # use the train set again each epoch to build prototypes
    reference_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Using {target_device} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    if unsupported_attributes:
        print(f"Attributes with fewer than two training classes: {', '.join(unsupported_attributes)}")

    model = Mobilenetv2(
        len(attribute_names), config.EMBEDDING_DIM, pretrained=pretrained
    ).to(target_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.METRIC_LEARNING_RATE)  # update backbone and heads

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)  # create the folder for the model weights

    best_val_score = -1.0
    history: list[dict[str, object]] = []
    attribute_width = max(map(len, ("Attribute", *attribute_names)))
    row_format = "{:<{width}}  {:>10}  {:>9}  {:>8}  {:>12}"

    for epoch in range(1, epochs + 1):
        loss, loss_by_index = train_metric_epoch(
            model, train_loader, optimizer, target_device,
            trainable_attribute_indices=trainable_attribute_indices,
        )
        train_loss_by_attribute = {
            name: loss_by_index.get(index) for index, name in enumerate(attribute_names)
        }

        # rebuild the prototypes with embeddings from the current model
        reference_vectors, reference_ids = collect_attribute_embeddings(
            model, reference_loader, target_device
        )
        prototypes = build_attribute_prototypes(reference_vectors, reference_ids, attribute_names)

        # reuse the training embeddings to calculate train accuracy
        train_accuracy, _ = evaluate_attributes(
            reference_vectors, reference_ids, prototypes, attribute_names
        )

        # score the validation images using the training prototypes
        val_vectors, val_ids = collect_attribute_embeddings(model, val_loader, target_device)
        val_accuracy, val_scores = evaluate_attributes(
            val_vectors, val_ids, prototypes, attribute_names
        )

        available = [score for score in val_scores.values() if score is not None]
        if not available:
            raise ValueError("验证集没有可用标注。")
        
        mean_val_f1 = sum(available) / len(available)
        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "train_loss_by_attribute": train_loss_by_attribute,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_scores,
        })

        print(
            f"\nEpoch {epoch:02d}/{epochs:02d} | Train loss {loss:.4f} | "
            f"Mean val Macro-F1 {mean_val_f1:.4f}"
        )
        print(row_format.format(
            "Attribute", "Train loss", "Train acc", "Val acc", "Val Macro-F1",
            width=attribute_width,
        ))
        for name in attribute_names:
            print(row_format.format(
                name,
                format_metric(train_loss_by_attribute[name], ".4f"),
                format_metric(train_accuracy[name], ".1%"),
                format_metric(val_accuracy[name], ".1%"),
                format_metric(val_scores[name], ".3f"),
                width=attribute_width,
            ))

        # save the model and prototypes when mean validation macro-f1 improves
        if mean_val_f1 > best_val_score:
            best_val_score = mean_val_f1
            checkpoint: MetricCheckpoint = {
                "kind": "ka_attribute_metric",
                "model_state_dict": model.state_dict(),
                "attribute_names": list(attribute_names),
                "attribute_value_sets": attribute_value_sets,
                "image_size": image_size,
                "embedding_dim": config.EMBEDDING_DIM,
                "prototypes": prototypes,
                "epoch": epoch,
            }
            torch.save(checkpoint, checkpoint_path)

    # load best checkpoint and evaluate once on the test set
    best_model, checkpoint = load_metric_model(checkpoint_path, target_device)
    test_vectors, test_ids = collect_attribute_embeddings(best_model, test_loader, target_device)
    test_predictions = collect_attribute_predictions(
        test_vectors, test_ids, checkpoint["prototypes"], attribute_names
    )
    test_accuracy, test_scores = summarize_attribute_predictions(test_predictions)
    test_class_accuracy, test_confusion_matrices = summarize_class_predictions(
        test_predictions, checkpoint["attribute_value_sets"]
    )

    # write the training history and the final test results to one json file
    report: dict[str, object] = {
        "seed": config.SEED,
        "pretrained": pretrained,
        "attribute_names": list(attribute_names),
        "unsupported_attributes": unsupported_attributes,
        "batch_size": batch_size,
        "learning_rate": config.METRIC_LEARNING_RATE,
        "margin": config.METRIC_MARGIN,
        "best_epoch": checkpoint["epoch"],
        "history": history,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_scores,
        "test_class_accuracy": test_class_accuracy,
        "test_confusion_matrices": test_confusion_matrices,
    }
    
    report_path = checkpoint_path.with_suffix(".history.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_test_results(
        checkpoint["epoch"], test_accuracy, test_scores, test_class_accuracy,
    )
    if outputs_dir is not None:
        paths = plot_test_confusion_matrices(test_confusion_matrices, outputs_dir)
        print(f"Saved {len(paths)} confusion matrices to {outputs_dir}")
    print(f"\nModel {checkpoint_path}\nTraining history {report_path}")


# run training with the settings in config
def main() -> None:
    train_metric_model(
        csv_path=config.TRAIN_CSV_PATH,
        checkpoint_path=config.METRIC_MODEL_PATH,
        epochs=config.EPOCHS,
        target_device=torch.device(config.device),
        pretrained=config.USE_PRETRAINED_WEIGHTS,
        outputs_dir=config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
