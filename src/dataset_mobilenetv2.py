"""Data loading and shared preprocessing for ka glyph images."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from config import SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


""" class coresspondent to a data record in csv """
@dataclass(frozen=True)
class AttributeSample:
    path: Path  # image file path
    values: tuple[str | None, ...]  # attributs


# match csv rows to image paths and read the selected attributes
def read_attribute_samples(
    csv_path: Path, attribute_names: tuple[str, ...]
) -> list[AttributeSample]:
    
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # check for the image name column and all requested attributes
    required = {"Deno", *attribute_names}
    missing_columns = required.difference(table.columns)
    if missing_columns:
        raise ValueError(f"CSV dosen't has ：{sorted(missing_columns)}")

    samples: list[AttributeSample] = []
    seen_paths: set[Path] = set()   # image that we already checked
    missing_files = 0

    # read each csv row as a dict
    for row in table.to_dict("records"):
        deno: str = row["Deno"].strip()
        if not deno:
            continue

        # keep the attribute values in the same order as attribute_names
        cleaned_values: list[str | None] = []
        for name in attribute_names:
            value: str = row[name].strip()
            if value.upper() in {"", "NA"}:
                cleaned_values.append(None)  # set blank or NA values to None
            else:
                cleaned_values.append(value)
        values = tuple(cleaned_values)
        if all(value is None for value in values):
            continue  # skip images with no known attributes

        filename: str = f"{deno}.png"

        # look for the png in the same directory as the csv
        path: Path = (csv_path.parent / filename).resolve()
        if not path.is_file():
            missing_files += 1
            continue  # skip this row if the image file is missing
        if path in seen_paths:
            raise ValueError(f"one image has multi records in csv:{path}")
        seen_paths.add(path)

        samples.append(AttributeSample(path, values))
    print(f"match {len(samples)} image with vailable attributs, skip{missing_files} images")

    return samples


# split the samples into train, val and test sets
def split_attribute_samples(
    samples: list[AttributeSample], *, seed: int = SEED
) -> tuple[list[AttributeSample], list[AttributeSample], list[AttributeSample]]:

    train_samples, remaining_samples = train_test_split(
        samples,
        test_size= 1 - TRAIN_RATIO,
        random_state=seed,
    )

    # get the test fraction within the samples left after the train split
    relative_test_size: float = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_samples, test_samples = train_test_split(
        remaining_samples,
        test_size=relative_test_size,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


# get the sorted values for each attribute, the list index is the label id
def build_attribute_value_sets(
    samples: list[AttributeSample], num_attributes: int
) -> list[list[str]]:
    attribute_value_set: list[list[str]] = []
    for index in range(num_attributes):
        values: set = {sample.values[index] for sample in samples}
        # skip missing values so they do not get a class id
        attribute_value_set.append(sorted(value for value in values if value is not None))
    return attribute_value_set


MOBILENET_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# load one image and convert it to a mobilenet input tensor
def load_mobilenet_image(path: Path, image_size: tuple[int, int]) -> Tensor:
    with Image.open(path) as original:
        # keep the aspect ratio and pad with gray to reach image_size
        image = ImageOps.pad(
            ImageOps.exif_transpose(original).convert("RGB"),
            image_size,
            method=Image.Resampling.LANCZOS,
            color=(128, 128, 128),
        )
    return MOBILENET_TRANSFORM(image)


""" custom dataset class"""
class AttributeDataset(Dataset):
    # save the sample paths and encode their attribute values
    def __init__(
        self, samples: list[AttributeSample], attribute_value_set: list[list[str]],
        image_size: tuple[int, int],
    ) -> None:
        self.samples = samples
        self.image_size = image_size

        # build one value-to-id mapping for each attribute
        mappings = [{value: i for i, value in enumerate(values)} for values in attribute_value_set]

        # use -1 for any missing attribute value
        self.attribute_ids = torch.tensor([
            [
                mapping[value] if value is not None else -1
                for mapping, value in zip(mappings, sample.values, strict=True)
            ]
            for sample in samples
        ], dtype=torch.long)

    # return the number of samples in this dataset
    def __len__(self) -> int:
        return len(self.samples)

    # load one image and return it with its attribute ids
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image: Tensor = load_mobilenet_image(self.samples[index].path, self.image_size)
        return image, self.attribute_ids[index]
