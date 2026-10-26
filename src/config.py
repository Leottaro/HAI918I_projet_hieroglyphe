"""Shared configuration for the Glyphnet experiment."""

from pathlib import Path
import torch


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data" / "glyphnet_data" / "Preprocessed"

# Image dimension [height, width]
IMAGE_SIZE: tuple[int, int] = (75, 50)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

SEED: int = 42

TOP_K_CLASSES = 40  # the top k classes that we want to classify

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

BATCH_SIZE: int = 32

LEARNING_RATE: float = 1e-3 

EPOCHS = 30
