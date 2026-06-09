from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import torch
from torch import nn

from src.cnn_edge_extractor.models.unet import UNet


def compute_slope_degrees(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    dem = np.nan_to_num(dem, nan=0.0).astype(np.float32)
    dz_dy, dz_dx = np.gradient(dem, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return np.nan_to_num(slope_deg, nan=0.0, posinf=0.0, neginf=0.0)


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = UNet(in_channels=1, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def normalize_patch_zscore(patch: np.ndarray) -> np.ndarray:
    patch = np.nan_to_num(patch, nan=0.0).astype(np.float32)
    mean = patch.mean()
    std = patch.std()
    if std > 1e-6:
        patch = (patch - mean) / std
    return patch.astype(np.float32)


def make_weight_window(tile_size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:tile_size, 0:tile_size]
    cy, cx = (tile_size - 1) / 2.0, (tile_size - 1) / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = (tile_size / 2.0) ** 2
    return np.exp(-dist2 / (2.0 * sigma2)).astype(np.float32)


def sliding_window_predict(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int,
    stride: int,
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    weights = make_weight_window(tile_size)

    with torch.no_grad():
        rows = list(range(0, max(height - tile_size + 1, 1), stride))
        cols = list(range(0, max(width - tile_size + 1, 1), stride))

        if rows[-1] != height - tile_size:
            rows.append(height - tile_size)
        if cols[-1] != width - tile_size:
            cols.append(width - tile_size)

        rows = sorted(set(r for r in rows if r >= 0))
        cols = sorted(set(c for c in cols if c >= 0))

        for row in rows:
            for col in cols:
                patch = image[row:row + tile_size, col:col + tile_size]
                if patch.shape != (tile_size, tile_size):
                    continue

                patch = normalize_patch_zscore(patch)
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

                prob_map[row:row + tile_size, col:col + tile_size] += probs * weights
                weight_map[row:row + tile_size, col:col + tile_size] += weights

    valid = weight_map > 0
    prob_map[valid] /= weight_map[valid]
    prob_map[~valid] = 0.0
    mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, mask


def predict_slope_full_dem(
    dem_path: Path,
    checkpoint_path: Path,
    out_slope_path: Path,
    out_prob_path: Path,
    out_mask_path: Path,
    tile_size: int = 100,
    stride: int = 50,
    threshold: float = 0.30,
) -> None:
    dem_path = dem_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"DEM: {dem_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {threshold}")

    model = load_model(checkpoint_path, device)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem = np.nan_to_num(dem, nan=0.0)
        profile = src.profile.copy()
        res_x, res_y = src.res

    slope = compute_slope_degrees(dem, res_x=res_x, res_y=res_y)
    print(f"Slope raster shape: {slope.shape}")
    print(f"Slope range: min={slope.min():.4f}, max={slope.max():.4f}")

    prob_map, mask = sliding_window_predict(
        model=model,
        image=slope,
        tile_size=tile_size,
        stride=stride,
        threshold=threshold,
        device=device,
    )

    out_slope_path.parent.mkdir(parents=True, exist_ok=True)
    out_prob_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    slope_profile = profile.copy()
    slope_profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=0.0)

    prob_profile = profile.copy()
    prob_profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=0.0)

    mask_profile = profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)

    with rasterio.open(out_slope_path, "w", **slope_profile) as dst:
        dst.write(slope, 1)

    with rasterio.open(out_prob_path, "w", **prob_profile) as dst:
        dst.write(prob_map, 1)

    with rasterio.open(out_mask_path, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    print(f"Saved slope raster to:     {out_slope_path}")
    print(f"Saved probability map to: {out_prob_path}")
    print(f"Saved binary mask to:     {out_mask_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    dem_path = project_root / "data" / "raw" / "1m" / "grib_1m.tif"
    checkpoint_path = project_root / "models" / "checkpoints_slope" / "best.ckpt"
    out_slope_path = project_root / "outputs" / "predictions" / "grib_1m_slope.tif"
    out_prob_path = project_root / "outputs" / "predictions" / "grib_1m_slope_prob.tif"
    out_mask_path = project_root / "outputs" / "predictions" / "grib_1m_slope_mask_t06.tif"

    print(f"Project root: {project_root}")
    print(f"DEM exists: {dem_path.exists()} -> {dem_path}")
    print(f"Checkpoint exists: {checkpoint_path.exists()} -> {checkpoint_path}")

    predict_slope_full_dem(
        dem_path=dem_path,
        checkpoint_path=checkpoint_path,
        out_slope_path=out_slope_path,
        out_prob_path=out_prob_path,
        out_mask_path=out_mask_path,
        tile_size=100,
        stride=50,
        threshold=0.6,
    )