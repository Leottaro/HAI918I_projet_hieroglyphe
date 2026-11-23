from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import KA_DATA_DIR, PROJECT_ROOT, SEED


MAX_REGION_FRACTION: float = 0.80
MIN_REGION_FRACTION: float = 0.10
POINTS_PER_REGION: int = 5
DEFAULT_OUTPUT_SIZE: tuple[int, int] = (160, 160)
SAM_MODEL_CONFIG: str = "configs/sam2.1/sam2.1_hiera_s.yaml"


""" create SAM predictor with weights exist"""
def create_sam_predictor(device_name: str = "cpu") -> SAM2ImagePredictor:
    model_path: Path = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"SAM2 model file not found: {model_path}")

    sam_model: SAM2Base = build_sam2(
        config_file=SAM_MODEL_CONFIG,
        ckpt_path=str(model_path),
        device=device_name,
        mode="eval",
    )
    return SAM2ImagePredictor(sam_model)

"""
return binary image cleaned,
connection component
and random sample point in component
"""
def find_prompt_groups(
    rgb: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray], list[np.ndarray]]:
    
    gray: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, light_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    dark_binary: np.ndarray = 255 - light_binary
    clean_light: np.ndarray = np.zeros_like(light_binary)
    clean_dark: np.ndarray = np.zeros_like(dark_binary)
    groups: list[np.ndarray] = []

    min_area: float = gray.size * MIN_REGION_FRACTION
    max_area: float = gray.size * MAX_REGION_FRACTION
    for binary, clean_binary in (
        (light_binary, clean_light),
        (dark_binary, clean_dark),
    ):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # 0 as background, don't sample
        for region_id in range(1, num_labels):
            area: int = int(stats[region_id, cv2.CC_STAT_AREA])
            if not min_area <= area <= max_area:
                continue

            region_mask: np.ndarray = labels == region_id
            clean_binary[region_mask] = 255
            ys, xs = np.nonzero(region_mask)
            indices: np.ndarray = rng.choice(
                xs.size, size=min(POINTS_PER_REGION, xs.size), replace=False
            )
            
            points: np.ndarray = np.column_stack(
                (xs[indices], ys[indices])
            ).astype(np.float32)
            groups.append(points)

    return (clean_light, clean_dark), groups

""" segement image by current prompt, return a mask(bool) with the same size of original iamge"""
@torch.inference_mode()
def segment_image(
    predictor: SAM2ImagePredictor,
    rgb: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if len(point_coords) != len(point_labels):
        raise ValueError("Each prompt point must have a label")
    if len(point_coords) == 0 and box is None:
        raise ValueError("Please add points or a box")

    box_array: np.ndarray | None = (
        None if box is None else np.array(box, dtype=np.float32)
    )

    predictor.set_image(rgb)
    masks, _, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        box=box_array,
        multimask_output=False,
    )
    return masks[0].astype(bool)


""" calculate bounding box for foreground in mask"""
def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


"""
cut original image by current box,
resize and fill padding with gaussian texture
"""
def prepare_processed_image(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
    rng: np.random.Generator,
    noise_std: float = 6.0,
) -> Image.Image:
    left, top, right, bottom = crop_box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("Crop box must be a non-empty region inside the image")
    target_width, target_height = output_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Output dimensions must be positive")
    if noise_std < 0:
        raise ValueError("Background noise standard deviation cannot be negative")

    crop: Image.Image = image.convert("RGB").crop(crop_box)
    scale: float = min(target_width / crop.width, target_height / crop.height)
    # evit 0 size
    resized_width: int = min(target_width, max(1, round(crop.width * scale)))
    resized_height: int = min(target_height, max(1, round(crop.height * scale)))
    resized: Image.Image = crop.resize(
        (resized_width, resized_height), resample=Image.Resampling.LANCZOS
    )

    result: Image.Image = _create_gaussian_background(crop, output_size, rng, noise_std)
    offset_x: int = (target_width - resized_width) // 2
    offset_y: int = (target_height - resized_height) // 2
    result.paste(resized, (offset_x, offset_y))
    return result


""" return a gaussin texture for this image (mean + vriance)"""
def _create_gaussian_background(
    crop: Image.Image,
    output_size: tuple[int, int],
    rng: np.random.Generator,
    noise_std: float,
) -> Image.Image:
    pixels: np.ndarray = np.array(crop)
    border_pixels: np.ndarray = np.concatenate(
        (pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]),
        axis=0,
    )
    background_color: np.ndarray = np.median(border_pixels, axis=0)
    width, height = output_size
    # all channel shoule be changed together
    noise: np.ndarray = rng.normal(0.0, noise_std, size=(height, width, 1))
    background_pixels: np.ndarray = np.clip(
        background_color + noise, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(background_pixels)


def main() -> None:
    with Image.open(KA_DATA_DIR / "KIU_556.png") as image:
        rgb: np.ndarray = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    rng: np.random.Generator = np.random.default_rng(SEED)
    _, groups = find_prompt_groups(rgb, rng)
    preview: np.ndarray = rgb.copy()
    for points in groups:
        for x, y in points:
            cv2.circle(preview, (int(x), int(y)), 3, (0, 255, 0), thickness=-1)
    Image.fromarray(preview).show()


if __name__ == "__main__":
    main()
