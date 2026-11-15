import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, RandomAffine, ToTensor
import torchvision.transforms.functional as TF
import albumentations as A

from sklearn.model_selection import train_test_split

from config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
    AUG_MAX_SHIFT_PERCENT, AUG_MAX_ROTATION_DEG, AUG_ZOOM_RANGE, BALANCE_CAP_FACTOR,
)

def class_name_from_path(image_path: Path) -> str:
    return image_path.stem.rsplit("_", 1)[-1]

def build_class_to_idx(selected_classes: list[str]) -> dict[str, int]:
    if not selected_classes:
        raise ValueError("selected classes is empty.")

    sorted_class_names: list[str] = sorted(set(selected_classes))

    class_to_idx: dict[str, int] = {}

    for index, class_name in enumerate(sorted_class_names):
        class_to_idx[class_name] = index

    return class_to_idx

RANDOM_AFFINE: A.BasicTransform = A.Affine(scale=AUG_ZOOM_RANGE, translate_percent=(-AUG_MAX_SHIFT_PERCENT, AUG_MAX_SHIFT_PERCENT), rotate=(-AUG_MAX_ROTATION_DEG, AUG_MAX_ROTATION_DEG))
# random shift + rotation + zoom (Barucci et al. 2021, Sec. II-B.3)
def random_shift_rotate_zoom(image: Image.Image) -> Image.Image:
    return Image.fromarray(RANDOM_AFFINE(image=np.asarray(image))["image"])

# One training example = a source image + the operations to apply when it is loaded.
@dataclass(frozen=True)
class Sample:
    path: Path
    jitter: bool = False  # random shift / rotation / zoom
    flip: bool = False    # horizontal mirror

# custom dataset class for pytorch
class HieroglyphDataset(Dataset):

    def __init__(self, samples: list[Sample], class_to_idx: dict[str, int]):
        super().__init__()
        self.samples: list[Sample] = samples
        self.class_to_idx: dict[str, int] = class_to_idx
        self.to_tensor: ToTensor = ToTensor()


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample: Sample = self.samples[index]
        label: int = self.class_to_idx[class_name_from_path(sample.path)]

        with Image.open(sample.path) as file:
            image: Image.Image = file.convert("L")

        if sample.jitter:
            image = random_shift_rotate_zoom(image)

        if sample.flip:
            image = ImageOps.mirror(image)

        return self.to_tensor(image), label


# Training-set balancing + augmentation, as in the paper:
#   1. classes with more than  BALANCE_CAP_FACTOR * mean  images are randomly downsampled
#   2. classes with fewer than mean images are topped up to the mean with
#      randomly shifted / rotated / zoomed copies
#   3. every image of the resulting set also gets a horizontally flipped copy
def build_training_samples(train_paths: list[Path], seed: int = SEED) -> list[Sample]:
    paths_by_class: dict[str, list[Path]] = defaultdict(list)

    for image_path in train_paths:
        paths_by_class[class_name_from_path(image_path)].append(image_path)

    mean_count: float = len(train_paths) / len(paths_by_class)
    upper_limit: int = int(BALANCE_CAP_FACTOR * mean_count)
    lower_limit: int = int(round(mean_count))

    rng = random.Random(seed)
    samples: list[Sample] = []

    for class_name in sorted(paths_by_class):
        class_paths: list[Path] = sorted(paths_by_class[class_name])

        if len(class_paths) > upper_limit:
            class_paths = rng.sample(class_paths, upper_limit)

        samples.extend(Sample(path) for path in class_paths)

        # cycle over the sources so that each one is reused about equally often
        missing: int = lower_limit - len(class_paths)
        for i in range(max(missing, 0)):
            samples.append(Sample(class_paths[i % len(class_paths)], jitter=True))

    num_balanced: int = len(samples)
    samples.extend(replace(sample, flip=True) for sample in list(samples))

    print(
        f"Training set: {len(train_paths)} images -> {num_balanced} after balancing "
        f"(cap {upper_limit}, floor {lower_limit}) -> {len(samples)} after horizontal flip"
    )

    return samples


# select top k classes from all class
def select_top_k_class(label_counts: Counter[str], top_k: int) -> list[str]:
    top_k_labels: list[str] = []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    known_counts: Counter[str] = label_counts.copy()
    known_counts.pop("UNKNOWN", None)


    if top_k > len(known_counts):
        raise ValueError("top_k exceeds the number of known classes.")

    for label, count in known_counts.most_common(top_k):
        top_k_labels.append(label)

    return top_k_labels


# get the image path from selected classes
def filter_image_paths(image_paths: list[Path], selected_classes: list[str]) -> list[Path]:
    selected_class_set: set[str] = set(selected_classes)

    selected_paths: list[Path] = []

    for image_path in image_paths:
        class_name: str = image_path.stem.rsplit("_", 1)[-1]

        if class_name in selected_class_set:
            selected_paths.append(image_path)

    return selected_paths


# split the dataset according settings in config.py
def split_dataset(image_paths: list[Path], seed: int = SEED,) -> tuple[list[Path], list[Path], list[Path]]:
    labels: list[str] = []

    for image_path in image_paths:
        class_name: str = image_path.stem.rsplit("_", 1)[-1]
        labels.append(class_name)

    train_paths: list[Path]
    remain_paths: list[Path]

    train_paths, remain_paths = train_test_split(
        image_paths,
        test_size = 1 - TRAIN_RATIO,
        random_state = seed,
        stratify = labels,
    )

    remaining_labels: list[str] = []

    for image_path in remain_paths:
        class_name = image_path.stem.rsplit("_", 1)[-1]
        remaining_labels.append(class_name)

    val_paths: list[Path]
    test_paths: list[Path]

    relative_test_ratio: float = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_paths, test_paths = train_test_split(
        remain_paths,
        test_size = relative_test_ratio,
        random_state = seed,
        stratify = remaining_labels,
    )
    
    return train_paths, val_paths, test_paths


# create a DataLoader for the given image paths for pytroch
def create_dataloader(image_paths: list[Path],
    class_to_idx: dict[str, int],
    batch_size: int,
    augment: bool,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if augment:
        samples: list[Sample] = build_training_samples(image_paths)
    else:
        samples = [Sample(image_path) for image_path in image_paths]

    image_dataset: HieroglyphDataset = HieroglyphDataset(samples, class_to_idx)

    dataloader: DataLoader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader