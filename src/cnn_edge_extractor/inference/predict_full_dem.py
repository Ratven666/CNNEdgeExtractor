from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import torch
from torch import nn

from src.cnn_edge_extractor.models.unet import UNet


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = UNet(in_channels=1, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sliding_window_predict(
    model: nn.Module,
    dem: np.ndarray,
    tile_size: int = 100,
    stride: int = 50,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет скользящее окно по DEM и возвращает:
    - prob_map: усреднённые вероятности [0..1]
    - mask: бинарная маска (0/1)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    height, width = dem.shape
    prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # простая центральная весовая маска: центр окна весит больше
    yy, xx = np.mgrid[0:tile_size, 0:tile_size]
    cy, cx = (tile_size - 1) / 2.0, (tile_size - 1) / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = (tile_size / 2.0) ** 2
    window_weights = np.exp(-dist2 / (2.0 * sigma2)).astype(np.float32)

    with torch.no_grad():
        for row in range(0, height - tile_size + 1, stride):
            for col in range(0, width - tile_size + 1, stride):
                patch = dem[row:row + tile_size, col:col + tile_size]
                if not np.isfinite(patch).any():
                    continue

                patch = np.nan_to_num(patch, nan=0.0).astype(np.float32)
                # нормализация на окно
                mean = patch.mean()
                std = patch.std()
                if std > 1e-6:
                    patch = (patch - mean) / std

                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().squeeze().numpy().astype(np.float32)

                prob_map[row:row + tile_size, col:col + tile_size] += probs * window_weights
                weight_map[row:row + tile_size, col:col + tile_size] += window_weights

    # избежать деления на ноль
    mask_nonzero = weight_map > 0
    prob_map[mask_nonzero] /= weight_map[mask_nonzero]
    prob_map[~mask_nonzero] = 0.0

    binary_mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, binary_mask


def predict_full_dem(
    dem_path: Path,
    checkpoint_path: Path,
    out_prob_path: Path,
    out_mask_path: Path,
    tile_size: int = 100,
    stride: int = 50,
    threshold: float = 0.5,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"DEM: {dem_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem = np.nan_to_num(dem, nan=0.0)
        profile = src.profile.copy()

    print(f"DEM shape: {dem.shape}")

    prob_map, binary_mask = sliding_window_predict(
        model=model,
        dem=dem,
        tile_size=tile_size,
        stride=stride,
        device=device,
        threshold=threshold,
    )

    out_prob_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    prob_profile = profile.copy()
    prob_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=0.0,
    )

    mask_profile = profile.copy()
    mask_profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(out_prob_path, "w", **prob_profile) as dst:
        dst.write(prob_map, 1)

    with rasterio.open(out_mask_path, "w", **mask_profile) as dst:
        dst.write(binary_mask, 1)

    print(f"Saved probability map to: {out_prob_path}")
    print(f"Saved binary mask to:     {out_mask_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    dem_path = project_root / "data" / "raw" / "1m" / "grib_1m.tif"
    checkpoint_path = project_root / "models" / "checkpoints" / "best.ckpt"
    out_prob_path = project_root / "data" / "predictions" / "grib_1m_prob.tif"
    out_mask_path = project_root / "data" / "predictions" / "grib_1m_mask.tif"

    predict_full_dem(
        dem_path=dem_path,
        checkpoint_path=checkpoint_path,
        out_prob_path=out_prob_path,
        out_mask_path=out_mask_path,
        tile_size=100,
        stride=50,
        threshold=0.5,
    )
