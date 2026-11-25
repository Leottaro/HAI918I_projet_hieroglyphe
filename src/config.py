"""Shared configuration for the Glyphnet experiment."""

from pathlib import Path
import hashlib
import json
import torch
import albumentations as A
import cv2

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "Image_DAO_Sith_Treated"
TRAINING_IMG_LOG_DIR: Path = PROJECT_ROOT / "Training_Image_DAO_Sith"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BEST_CONFIG_DIR = OUTPUT_DIR / "best"

AUGMENTATION_TRANSFORMS = {
    "all": A.Compose([
        A.Affine(
            p=1.,
            scale=(0.95, 1.05),
            translate_percent=(-0.1, 0.1),
            rotate=(-10.0, 10.0),
            keep_ratio=False,
            border_mode=cv2.BORDER_REPLICATE,
        ),
        A.GaussNoise(p=1., std_range=(0.025, 0.025)),
        A.RandomBrightnessContrast(p=1., brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25)),
    ]),
    # "affine": A.Affine(
    #     p=1.,
    #     scale=(0.95, 1.05),
    #     translate_percent=(-0.1, 0.1),
    #     rotate=(-10.0, 10.0),
    #     keep_ratio=False,
    #     border_mode=cv2.BORDER_REPLICATE,
    # ),
    # "noise": A.GaussNoise(p=1., std_range=(0.025, 0.025)),
    # "brightness_contrast": A.RandomBrightnessContrast(p=1., brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25)),
}

class Config(dict):
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    image_size: int = 128
    epochs: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    balance_augmentation_factor: float = 32.0
    
    def __init__(self) -> None:
        super().__init__(
            seed = self.seed,
            train_ratio = self.train_ratio,
            val_ratio = self.val_ratio,
            test_ratio = self.test_ratio,
            image_size = self.image_size,
            batch_size = self.batch_size,
            learning_rate = self.learning_rate,
            epochs = self.epochs,
            balance_augmentation_factor = self.balance_augmentation_factor
        )

    def getResizePadTransform(self) -> A.BasicTransform:
        return A.Compose([
            A.LongestMaxSize(
                p = 1.0,
                max_size=self.image_size,
            ),
            A.PadIfNeeded(
                p=1.,
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_REPLICATE,
                position="center",
            ),
        ])
    
    def getDirName(self) -> str:
        return hashlib.sha256(json.dumps(self, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

CONFIG: Config = Config()