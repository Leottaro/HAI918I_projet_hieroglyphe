"""Shared configuration for ka attribute metric learning."""

from pathlib import Path
import torch


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "Image_DAO_Sith"
TRAIN_CSV_PATH: Path = DATA_DIR / "Tableau_imagette_MB.csv"
METRIC_MODEL_PATH: Path = PROJECT_ROOT / "checkpoints" / "ka_metric_all_attributes" / "model.pt"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

DEVICE_OVERRIDE: str | None = None
if DEVICE_OVERRIDE is not None:
    device: str = DEVICE_OVERRIDE
elif torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"

SEED: int = 42

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

BATCH_SIZE: int = 32
EPOCHS: int = 17
USE_PRETRAINED_WEIGHTS: bool = True
METRIC_LEARNING_RATE: float = 1e-3

ATTRIBUTES: tuple[str, ...] = (
    "Type_monu",
    "Mains",
    "Forme_main",
    "Doigts",
    "Doigts_indiv",
    "Espacement doigts",
    "Nombre doigts",
    "Symétrie",
    "Pouces_distingues",
    "Angle des coudes",
    "Type de jointure",
    "Technique",
    "Pavois",
    "Couleur",
    "Bracelets",
    "Combinaison",
)
IMAGE_SIZE: tuple[int, int] = (160, 160)
EMBEDDING_DIM: int = 32
METRIC_MARGIN: float = 0.25
