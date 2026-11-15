from pathlib import Path
from collections import Counter

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split

import albumentations as A
import numpy as np

from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED

# build a uniform mapping from str label to int 
def build_class_to_idx(selected_classes: list[str]) -> dict[str, int]:
    if not selected_classes:
        raise ValueError("selected classes is empty.")

    sorted_class_names: list[str] = sorted(set(selected_classes))

    class_to_idx: dict[str, int] = {}

    for index, class_name in enumerate(sorted_class_names):
        class_to_idx[class_name] = index

    return class_to_idx

# custom dataset class for pytorch
class HieroglyphDataset(Dataset):

    def __init__(self, image_paths: list[Path], class_to_idx: dict[str, int], transforms: list[A.BasicTransform]) :
        super().__init__()
        self.image_paths: list[Path] = image_paths
        self.transforms: list[A.BasicTransform] = transforms
        self.class_to_idx: dict[str, int] = class_to_idx
        self.to_tensor: ToTensor = ToTensor()


    def __len__(self) -> int:
        return len(self.image_paths) * (1 + len(self.transforms))


    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        img_index = index % len(self.image_paths)
        transform_index = index // len(self.image_paths)
        image_path: Path = self.image_paths[img_index]
        class_name: str = image_path.stem.rsplit("_", 1)[-1]
        label: int = self.class_to_idx[class_name]

        with Image.open(image_path) as image:
            if transform_index > 0:
                image = Image.fromarray(self.transforms[transform_index-1](image=np.asarray(image))["image"])
                
            image_tensor: Tensor = self.to_tensor(image.convert("L"))
            return  image_tensor, label


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
    transforms = [
        A.RGBShift(p=1.),
        A.ChannelShuffle(p=1.),
        A.CLAHE(p=1.),
        A.RandomBrightnessContrast(p=1.),
        A.RandomGamma(p=1.),
        A.ImageCompression(p=1.),
        
        A.HorizontalFlip(p=1.),
        A.VerticalFlip(p=1.),
        A.Compose([A.HorizontalFlip(p=1.), A.VerticalFlip(p=1.)]), # = 180 rotation
        # A.RandomCrop(p=1.),
    ] if augment else []
    
    image_dataset: HieroglyphDataset = HieroglyphDataset(image_paths, class_to_idx, transforms)

    dataloader: DataLoader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader