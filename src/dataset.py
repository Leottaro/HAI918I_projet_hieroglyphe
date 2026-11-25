import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split

from config import *

def build_class_to_idx(selected_classes: list[str]) -> dict[str, int]:
    if not selected_classes:
        raise ValueError("selected classes is empty.")

    sorted_class_names: list[str] = sorted(set(selected_classes), key=str)

    class_to_idx: dict[str, int] = {}
    for index, class_name in enumerate(sorted_class_names):
        class_to_idx[class_name] = index

    return class_to_idx

# One training example = a source image + the operations to apply when it is loaded.
@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    augment: bool = False
    flip: bool = False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented
        return (self.path, self.label, self.augment, self.flip) < (other.path, other.label, other.augment, other.flip)

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
        img_test_path: Path = TRAINING_IMG_LOG_DIR / sample.path.name
        save_image: bool = False
        
        TRAINING_IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        with Image.open(sample.path) as file:
            image = file.convert("L")
            
        if sample.flip:
            image = ImageOps.mirror(image)
            save_image = True
            img_test_path = img_test_path.with_name(f"{img_test_path.stem}_flip{img_test_path.suffix}")
        
        if sample.augment:
            transform_name = random.choice(list(AUGMENTATION_TRANSFORMS.keys()))
            image = Image.fromarray(AUGMENTATION_TRANSFORMS[transform_name](image=np.asarray(image))["image"])
            save_image = True
            img_test_path = img_test_path.with_name(f"{img_test_path.stem}_{transform_name}{img_test_path.suffix}")

        # Resize proportionally, then center the image on a black canvas.
        image = Image.fromarray(CONFIG.getResizePadTransform()(image=np.asarray(image))["image"])
        
        if save_image and not img_test_path.exists():
            TRAINING_IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
            image.save(img_test_path)

        return self.to_tensor(image), self.class_to_idx[sample.label]


def augment_training_samples(samples: list[Sample]) -> list[Sample]:
    samples_by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        samples_by_class[sample.label].append(sample)

    max_class_count: int = 0
    min_class_count: int = math.inf
    for samples in samples_by_class.values():
        count = len(samples)
        max_class_count = max(max_class_count, count)
        min_class_count = min(min_class_count, count)
    final_class_count = int(min_class_count + (max_class_count - min_class_count) * CONFIG.max_augmentation_percent)

    augment_samples: list[Sample] = []
    for class_name in sorted(samples_by_class):
        augment_samples.extend(sorted(samples_by_class[class_name]))
        current_class_count = len(samples_by_class[class_name])
        if current_class_count > final_class_count:
            augment_samples = augment_samples[:final_class_count]
        elif current_class_count < final_class_count:
            for i in range(current_class_count, final_class_count):
                sample = samples_by_class[class_name][i % current_class_count]
                augment_samples.append(Sample(sample.path, sample.label, augment=True))

    augment_flip_samples: list[Sample] = list(augment_samples)
    augment_flip_samples.extend(Sample(sample.path, sample.label, augment=sample.augment, flip=True) for sample in list(augment_flip_samples))
    
    return augment_flip_samples

# split the dataset according settings in config.py
def split_dataset(image_paths: list[Path], labels: list[str]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_paths: list[Path]
    remain_paths: list[Path]
    train_labels: list[str]
    remain_labels: list[str]

    train_paths, remain_paths = train_test_split(
        image_paths,
        test_size = 1 - CONFIG.train_ratio,
        random_state = CONFIG.seed,
        stratify = labels,
    )
    train_labels, remain_labels = train_test_split(
        labels,
        test_size = 1 - CONFIG.train_ratio,
        random_state = CONFIG.seed,
        stratify = labels,
    )

    val_paths: list[Path]
    test_paths: list[Path]
    val_labels: list[Path]
    test_labels: list[Path]

    relative_test_ratio: float = CONFIG.test_ratio / (CONFIG.val_ratio + CONFIG.test_ratio)

    val_paths, test_paths = train_test_split(
        remain_paths,
        test_size = relative_test_ratio,
        random_state = CONFIG.seed,
        stratify = remain_labels,
    )
    val_labels, test_labels = train_test_split(
        remain_labels,
        test_size = relative_test_ratio,
        random_state = CONFIG.seed,
        stratify = remain_labels,
    )
    
    return [Sample(train_paths[i], train_labels[i]) for i in range(len(train_labels))], [Sample(val_paths[i], val_labels[i]) for i in range(len(val_labels))], [Sample(test_paths[i], test_labels[i]) for i in range(len(test_labels))]


# create a DataLoader for the given image paths for pytroch
def create_dataloader(
    samples: list[Sample],
    class_to_idx: dict[str, int],
    batch_size: int,
    augment: bool,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if augment:
        samples = augment_training_samples(samples)

    image_dataset: HieroglyphDataset = HieroglyphDataset(samples, class_to_idx)

    dataloader: DataLoader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader

def read_dataset(image_paths: list[Path], labels: list[str]) -> tuple[dict[str, int], DataLoader, DataLoader, DataLoader, DataLoader]:
    # label_counts: Counter[str] = inspect_images(image_paths)
    label_counts: Counter[str] = Counter()
    for label in labels:
        label_counts[label] += 1
    
    class_to_idx: dict[str, int] = build_class_to_idx(labels)
    
    train_samples, val_samples, test_samples = split_dataset(
        image_paths,
        labels,
    )
    train_loader: DataLoader = create_dataloader(
        samples=train_samples,
        class_to_idx=class_to_idx,
        batch_size=CONFIG.batch_size,
        augment=True,
        shuffle=True,
        num_workers=0
    )
    train_eval_loader: DataLoader = create_dataloader(
        samples=train_samples,
        class_to_idx=class_to_idx,
        batch_size=CONFIG.batch_size,
        augment=False,
        shuffle=False,
        num_workers=0,
    )
    val_loader: DataLoader = create_dataloader(
        samples=val_samples,
        class_to_idx=class_to_idx,
        batch_size=CONFIG.batch_size,
        augment=False,
        shuffle=False,
        num_workers=0
    )
    test_loader: DataLoader = create_dataloader(
        samples=test_samples,
        class_to_idx=class_to_idx,
        batch_size=CONFIG.batch_size,
        augment=False,
        shuffle=False,
        num_workers=0
    )
    return [class_to_idx,train_loader,train_eval_loader,val_loader,test_loader]