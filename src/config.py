"""Shared configuration for the Glyphnet experiment."""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data" / "glyphnet_data" / "Preprocessed"

# Image dimension [height, width]
IMAGE_SIZE: tuple[int, int] = (75, 50)

SEED: int = 42

TOP_K_CLASSES = 40  # the top k classes that we want to classify

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15