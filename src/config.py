"""Shared configuration for the Glyphnet experiment."""

from pathlib import Path
import torch


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "Image_DAO_Sith"
TRAINING_IMG_LOG_DIR: Path = PROJECT_ROOT / "Training_Image_DAO_Sith"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

SEED: int = 42

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

BATCH_SIZE: int = 32

LEARNING_RATE: float = 1e-3 

EPOCHS = 30

AUG_MAX_SHIFT_PERCENT: float = 10.0
AUG_MAX_ROTATION_DEG: float = 10.0
AUG_ZOOM_RANGE: tuple[float, float] = (0.95, 1.05)
BALANCE_CAP_FACTOR: float = 2.0