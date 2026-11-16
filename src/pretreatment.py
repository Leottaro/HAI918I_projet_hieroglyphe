import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import KA_DATA_DIR, PROJECT_ROOT, SEED

MAX_REGION_FRACTION: float = 0.80
MIN_REGION_FRACTION: float = 0.03
POINTS_PER_REGION: int = 5
MAX_MASK_FRACTION: float = 0.85
NMS_IOU: float = 0.80
DILATE_ITERATIONS: int = 0
TARGET_SIZE: int = 160
BORDER: int = 8

# create sam predictor from exist weights file
def create_sam_predictor(
    device_name: str = "cpu",
) -> SAM2ImagePredictor:
    model_path: Path = (
        PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"SAM2 model file not found: {model_path}"
        )

    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam_model: SAM2Base = build_sam2(
        config_file=model_config,
        ckpt_path=str(model_path),
        device=device_name,
        mode="eval",
    )

    predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam_model)
    return predictor


def find_prompt_groups(
        rgb: np.ndarray,
        rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray], list[np.ndarray]]:
    # transform rgb image to gray
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # segment gray image into light and dark part
    _, light_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark_binary = 255 - light_binary

    # create empty binary after cleaning to stock result
    clean_light = np.zeros_like(light_binary)   # zeros_like: create an array with same shape fill 0
    clean_dark = np.zeros_like(dark_binary)

    groups: list[np.ndarray] = []   # stock all components groups

    max_region_area: float = gray.size * MAX_REGION_FRACTION
    min_region_area: float = gray.size * MIN_REGION_FRACTION

    for binary, clean_binary in (
        (light_binary, clean_light),
        (dark_binary, clean_dark),
    ):
        # get numbre of components, label(nb of components, same shape with image), and stats(location, width, height, surface of components)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for region_id in range(1, num_labels):
            area: int = int(stats[region_id, cv2.CC_STAT_AREA]) # get area of a region

            # drop out areas too small or too big
            if area < min_region_area or area > max_region_area:
                continue

            # set true if label == this region, region_mask has the same shape with labels & image
            region_mask = (labels == region_id)

            # set this region to result binary
            clean_binary[region_mask] = 255

            # pixel location in region mask not false
            ys, xs = np.nonzero(region_mask)

            # select some random indeces of points in region
            indices = rng.choice(
                xs.size,
                size=min(POINTS_PER_REGION, xs.size),
                replace=False,
            )

            # points selected randomly
            points = np.column_stack((xs[indices], ys[indices])).astype(np.float32)

            # add point into group for this region
            groups.append(points)

    return (clean_light, clean_dark), groups


# use SAM to segement image by prompt points and box
@torch.inference_mode()
def segment_image(
    predictor: SAM2ImagePredictor,
    rgb: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> np.ndarray:

    # convert box to np array
    box_array: np.ndarray | None = (
        None
        if box is None
        else np.array(box, dtype=np.float32)
    )

    predictor.set_image(rgb)

    masks: np.ndarray

    # use prompt points and box to get segment result mask
    masks, _, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        box=box_array,
        multimask_output=False,
    )

    mask: np.ndarray = masks[0].astype(bool)
    return mask


def mask_to_box(
        mask: np.ndarray
) -> tuple[int, int, int, int] | None:
    ys: np.ndarray
    xs: np.ndarray
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        return None

    left: int = int(xs.min())
    top: int = int(ys.min())

    right: int = int(xs.max()) + 1
    bottom: int = int(ys.max()) + 1

    return left, top, right, bottom

    

def main() -> None:
    image_path: Path = KA_DATA_DIR / "KIU_556.png"

    with Image.open(image_path) as image:
        rgb_image: Image.Image = ImageOps.exif_transpose(image).convert("RGB")
        rgb: np.ndarray = np.array(rgb_image)

    rng: np.random.Generator = np.random.default_rng(SEED)

    (clean_light, clean_dark), groups = find_prompt_groups(rgb, rng)

    preview: np.ndarray = rgb.copy()

    # use opencv to draw prompt points
    for points in groups:
        for x, y in points:
            center: tuple[int, int] = (int(x), int(y))

            cv2.circle(
                preview,
                center,
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )
            
    # convert numpy array to pillow image and show
    preview_image: Image.Image = Image.fromarray(preview)
    preview_image.show()

    predictor: SAM2ImagePredictor = create_sam_predictor()

    print(f"SAM2 loaded on: {predictor.device}")
   

if __name__ == "__main__":

    main()
