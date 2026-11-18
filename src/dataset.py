import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, RandomAffine, ToTensor
import torchvision.transforms.functional as TF
import albumentations as A
import cv2

from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE, TRAIN_RATIO, TRAINING_IMG_LOG_DIR, VAL_RATIO, TEST_RATIO, SEED,
    AUG_MAX_SHIFT_PERCENT, AUG_MAX_ROTATION_DEG, AUG_ZOOM_RANGE, BALANCE_CAP_FACTOR,
)

def build_class_to_idx(selected_classes: list[str]) -> dict[str, int]:
    if not selected_classes:
        raise ValueError("selected classes is empty.")

    sorted_class_names: list[str] = sorted(set(selected_classes), key=str)

    class_to_idx: dict[str, int] = {}
    for index, class_name in enumerate(sorted_class_names):
        class_to_idx[class_name] = index

    return class_to_idx

def get_min_size(images_path: list[Path]) -> int:
    min_width: int = math.inf
    min_height: int = math.inf
    for path in images_path:
        with Image.open(path) as image:
            min_width = min(min_width, image.size[0])
            min_height = min(min_height, image.size[1])
    return max(min_width, min_height)

# Avoid Albumentations' default zero (black) border when translating/rotating.
RANDOM_AFFINE: A.BasicTransform = A.Affine(
    p=1.,
    scale=AUG_ZOOM_RANGE,
    translate_percent=(-AUG_MAX_SHIFT_PERCENT, AUG_MAX_SHIFT_PERCENT),
    rotate=(-AUG_MAX_ROTATION_DEG, AUG_MAX_ROTATION_DEG),
    keep_ratio=False,
    border_mode=cv2.BORDER_REPLICATE,
)
# random shift + rotation + zoom (Barucci et al. 2021, Sec. II-B.3)
def random_shift_rotate_zoom(image: Image.Image) -> Image.Image:
    transformed = RANDOM_AFFINE(image=np.asarray(image.convert("RGB")))
    return Image.fromarray(transformed["image"])

# One training example = a source image + the operations to apply when it is loaded.
@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    jitter: bool = False  # random shift / rotation / zoom
    flip: bool = False    # horizontal mirror

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented
        return (self.path, self.label, self.jitter, self.flip) < (other.path, other.label, other.jitter, other.flip)

# custom dataset class for pytorch
class HieroglyphDataset(Dataset):

    def __init__(self, samples: list[Sample], class_to_idx: dict[str, int], image_size: int):
        super().__init__()
        self.samples: list[Sample] = samples
        self.class_to_idx: dict[str, int] = class_to_idx
        self.image_size: int = image_size
        self.to_tensor: ToTensor = ToTensor()


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample: Sample = self.samples[index]
        img_test_path: Path = TRAINING_IMG_LOG_DIR / sample.path.name
        save_image: bool = False
        
        with Image.open(sample.path) as file:
            image = file.convert("L")

        if sample.flip:
            image = ImageOps.mirror(image)
            save_image = True
            img_test_path = img_test_path.with_name(
                f"{img_test_path.stem}_flip{img_test_path.suffix}"
            )
        
        if sample.jitter:
            image = random_shift_rotate_zoom(image)
            save_image = True
            img_test_path = img_test_path.with_name(
                f"{img_test_path.stem}_jitter{img_test_path.suffix}"
            )


        # Resize proportionally, then center the image on a black canvas.
        image = ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
            color=None,
        )
        
        if save_image and not img_test_path.exists():
            TRAINING_IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
            image.save(img_test_path)

        return self.to_tensor(image), self.class_to_idx[sample.label]


# Training-set balancing + augmentation, as in the paper:
#   1. classes with more than  BALANCE_CAP_FACTOR * mean  images are randomly downsampled
#   2. classes with fewer than mean images are topped up to the mean with
#      randomly shifted / rotated / zoomed copies
#   3. every image of the resulting set also gets a horizontally flipped copy
def augment_training_samples(samples: list[Sample], seed: int = SEED) -> list[Sample]:
    samples_by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        samples_by_class[sample.label].append(sample)

    mean_count: float = len(samples) / len(samples_by_class)
    upper_limit: int = int(BALANCE_CAP_FACTOR * mean_count)
    lower_limit: int = int(round(mean_count))

    rng = random.Random(seed)
    new_samples: list[Sample] = []

    for class_name in sorted(samples_by_class):
        class_samples: list[Sample] = sorted(samples_by_class[class_name])

        if len(class_samples) > upper_limit:
            class_samples = rng.sample(class_samples, upper_limit)

        new_samples.extend(Sample(sample.path, sample.label) for sample in class_samples)
        
        # cycle over the sources so that each one is reused about equally often
        missing: int = lower_limit - len(class_samples)
        for i in range(max(missing, 0)):
            sample = class_samples[i % len(class_samples)]
            new_samples.append(Sample(sample.path, sample.label, jitter=True))

    num_balanced: int = len(new_samples)
    new_samples.extend(replace(sample, flip=True) for sample in list(new_samples))

    print(
        f"Training set: {len(samples)} images -> {num_balanced} after balancing "
        f"(cap {upper_limit}, floor {lower_limit}) -> {len(new_samples)} after horizontal flip"
    )

    return new_samples

# split the dataset according settings in config.py
def split_dataset(image_paths: list[Path], labels: list[str], seed: int = SEED,) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_paths: list[Path]
    remain_paths: list[Path]
    train_labels: list[str]
    remain_labels: list[str]

    train_paths, remain_paths = train_test_split(
        image_paths,
        test_size = 1 - TRAIN_RATIO,
        random_state = seed,
        stratify = labels,
    )
    train_labels, remain_labels = train_test_split(
        labels,
        test_size = 1 - TRAIN_RATIO,
        random_state = seed,
        stratify = labels,
    )

    val_paths: list[Path]
    test_paths: list[Path]
    val_labels: list[Path]
    test_labels: list[Path]

    relative_test_ratio: float = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_paths, test_paths = train_test_split(
        remain_paths,
        test_size = relative_test_ratio,
        random_state = seed,
        stratify = remain_labels,
    )
    val_labels, test_labels = train_test_split(
        remain_labels,
        test_size = relative_test_ratio,
        random_state = seed,
        stratify = remain_labels,
    )
    
    return [Sample(train_paths[i], train_labels[i]) for i in range(len(train_labels))], [Sample(val_paths[i], val_labels[i]) for i in range(len(val_labels))], [Sample(test_paths[i], test_labels[i]) for i in range(len(test_labels))]


# create a DataLoader for the given image paths for pytroch
def create_dataloader(
    samples: list[Sample],
    class_to_idx: dict[str, int],
    batch_size: int,
    image_size: int,
    augment: bool,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if augment:
        samples = augment_training_samples(samples)

    image_dataset: HieroglyphDataset = HieroglyphDataset(samples, class_to_idx, image_size)

    dataloader: DataLoader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader

def read_dataset(image_paths: list[Path], labels: list[str], image_size: int) -> tuple[dict[str, int], DataLoader, DataLoader, DataLoader, DataLoader]:
    # label_counts: Counter[str] = inspect_images(image_paths)
    label_counts: Counter[str] = Counter()
    for label in labels:
        label_counts[label] += 1
    
    class_to_idx: dict[str, int] = build_class_to_idx(labels)
    
    train_samples, val_samples, test_samples = split_dataset(
        image_paths,
        labels,
        SEED,
    )
    train_loader: DataLoader = create_dataloader(
        samples=train_samples,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        augment=True,
        shuffle=True,
        num_workers=0
    )
    train_eval_loader: DataLoader = create_dataloader(
        samples=train_samples,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        augment=False,
        shuffle=False,
        num_workers=0,
    )
    val_loader: DataLoader = create_dataloader(
        samples=val_samples,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        augment=False,
        shuffle=False,
        num_workers=0
    )
    test_loader: DataLoader = create_dataloader(
        samples=test_samples,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        augment=False,
        shuffle=False,
        num_workers=0
    )
    return [class_to_idx,train_loader,train_eval_loader,val_loader,test_loader]