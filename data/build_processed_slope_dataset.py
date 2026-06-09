from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import shutil

import albumentations as A
import numpy as np
import rasterio


def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    raw_dir = raw_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    dem_files = sorted(raw_dir.glob("bd_res_1m_*.tif"))
    pairs: list[tuple[Path, Path]] = []

    for dem_path in dem_files:
        suffix = dem_path.name.replace("bd_res_1m_", "")
        mask_path = raw_dir / f"line_res_1m_{suffix}"

        if mask_path.exists():
            pairs.append((dem_path, mask_path))
        else:
            print(f"[WARN] Mask not found for {dem_path.name}")

    return pairs


def reset_processed_dir(base_dir: Path) -> None:
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ("train", "val", "test"):
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any], float, float]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        res_x, res_y = src.res
    return arr, profile, res_x, res_y


def write_raster(
    path: Path,
    arr: np.ndarray,
    profile: dict[str, Any],
    dtype: str | None = None,
    nodata: float | int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()

    if dtype is not None:
        out_profile["dtype"] = dtype

    out_profile["count"] = 1
    out_profile["compress"] = "lzw"
    out_profile["nodata"] = nodata

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr, 1)


def compute_slope_degrees(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    dem = np.nan_to_num(dem, nan=0.0).astype(np.float32)
    dz_dy, dz_dx = np.gradient(dem, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return np.nan_to_num(slope_deg, nan=0.0, posinf=0.0, neginf=0.0)


def get_train_augmentation() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5,
        ),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.GaussNoise(p=0.3),
    ])


def save_pair(
    dem_path: Path,
    mask_path: Path,
    split_dir: Path,
    sample_name: str,
    transform: A.Compose | None = None,
) -> None:
    dem_arr, dem_profile, res_x, res_y = read_raster(dem_path)
    mask_arr, mask_profile, _, _ = read_raster(mask_path)

    slope_arr = compute_slope_degrees(dem_arr, res_x=res_x, res_y=res_y)
    mask_arr = (np.nan_to_num(mask_arr, nan=0.0) > 0.5).astype(np.uint8)

    if transform is not None:
        transformed = transform(image=slope_arr, mask=mask_arr)
        slope_arr = transformed["image"].astype(np.float32)
        mask_arr = transformed["mask"].astype(np.uint8)

    write_raster(
        split_dir / "images" / f"{sample_name}.tif",
        slope_arr,
        dem_profile,
        dtype=rasterio.float32,
        nodata=0,
    )
    write_raster(
        split_dir / "masks" / f"{sample_name}.tif",
        mask_arr,
        mask_profile,
        dtype=rasterio.uint8,
        nodata=0,
    )


def build_processed_slope_dataset(
    raw_dir: Path,
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    augmentations_per_train_image: int = 20,
) -> None:
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()

    random.seed(seed)
    reset_processed_dir(processed_dir)

    pairs = find_pairs(raw_dir)
    if not pairs:
        raise ValueError(f"No paired TIFF files found in {raw_dir}")

    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"Raw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")
    print(f"Total pairs: {n_total}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    train_aug = get_train_augmentation()

    for idx, (dem_path, mask_path) in enumerate(train_pairs):
        sample_id = f"train_{idx:03d}"
        save_pair(dem_path, mask_path, processed_dir / "train", sample_id)

        for aug_idx in range(augmentations_per_train_image):
            save_pair(
                dem_path,
                mask_path,
                processed_dir / "train",
                f"{sample_id}_aug_{aug_idx:02d}",
                transform=train_aug,
            )

    for idx, (dem_path, mask_path) in enumerate(val_pairs):
        save_pair(dem_path, mask_path, processed_dir / "val", f"val_{idx:03d}")

    for idx, (dem_path, mask_path) in enumerate(test_pairs):
        save_pair(dem_path, mask_path, processed_dir / "test", f"test_{idx:03d}")

    train_images = len(list((processed_dir / "train" / "images").glob("*.tif")))
    train_masks = len(list((processed_dir / "train" / "masks").glob("*.tif")))
    val_images = len(list((processed_dir / "val" / "images").glob("*.tif")))
    test_images = len(list((processed_dir / "test" / "images").glob("*.tif")))

    print("Done.")
    print(f"Train images/masks: {train_images}/{train_masks}")
    print(f"Val images: {val_images}")
    print(f"Test images: {test_images}")
    print(f"Processed slope dataset created in: {processed_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    print(f"Project root: {project_root}")

    build_processed_slope_dataset(
        raw_dir=project_root / "data" / "raw" / "1m" / "100",
        processed_dir=project_root / "data" / "processed_slope",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        augmentations_per_train_image=50,
    )