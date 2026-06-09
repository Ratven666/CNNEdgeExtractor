from __future__ import annotations
from pathlib import Path
import csv, random, shutil
import numpy as np
import rasterio
from rasterio.windows import Window

def raster_to_binary_mask(label_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(label_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    mask = (np.nan_to_num(arr, nan=0.0) > 0.5).astype(np.uint8)
    profile.update(dtype=rasterio.uint8, nodata=0, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask, 1)
    return out_path

def crop_aligned_tiles(dem_path: Path, mask_path: Path, output_dir: Path, tile_size: int = 100, stride: int = 100):
    images_dir = output_dir / 'tiles' / 'images'
    masks_dir = output_dir / 'tiles' / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with rasterio.open(dem_path) as dem_src, rasterio.open(mask_path) as mask_src:
        assert dem_src.width == mask_src.width and dem_src.height == mask_src.height, 'DEM and mask must have same shape'
        counter = 0
        for row in range(0, dem_src.height - tile_size + 1, stride):
            for col in range(0, dem_src.width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                dem = dem_src.read(1, window=window).astype(np.float32)
                mask = mask_src.read(1, window=window).astype(np.uint8)
                if np.all(~np.isfinite(dem)):
                    continue
                transform = dem_src.window_transform(window)
                dem_profile = dem_src.profile.copy()
                dem_profile.update(height=tile_size, width=tile_size, transform=transform, count=1, dtype=rasterio.float32, compress='lzw')
                mask_profile = mask_src.profile.copy()
                mask_profile.update(height=tile_size, width=tile_size, transform=transform, count=1, dtype=rasterio.uint8, nodata=0, compress='lzw')
                dem_out = images_dir / f'tile_{counter:04d}.tif'
                mask_out = masks_dir / f'tile_{counter:04d}.tif'
                with rasterio.open(dem_out, 'w', **dem_profile) as dst: dst.write(dem, 1)
                with rasterio.open(mask_out, 'w', **mask_profile) as dst: dst.write(mask, 1)
                records.append({'tile_id': counter, 'image_path': str(dem_out), 'mask_path': str(mask_out)})
                counter += 1
    return records

def split_tiles(records: list[dict], output_dir: Path, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed); recs = list(records); rng.shuffle(recs)
    n = len(recs); n_train = int(n * train_ratio); n_val = int(n * val_ratio); split_map = {}
    for idx, rec in enumerate(recs):
        split_map[rec['tile_id']] = 'train' if idx < n_train else 'val' if idx < n_train + n_val else 'test'
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for rec in records:
        split = split_map[rec['tile_id']]
        image_src = Path(rec['image_path']); mask_src = Path(rec['mask_path'])
        image_dst = output_dir / split / 'images' / image_src.name
        mask_dst = output_dir / split / 'masks' / mask_src.name
        shutil.copy2(image_src, image_dst); shutil.copy2(mask_src, mask_dst)
        manifest_rows.append({'tile_id': rec['tile_id'], 'split': split, 'image_path': str(image_dst), 'mask_path': str(mask_dst)})
    manifest = output_dir / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tile_id', 'split', 'image_path', 'mask_path'])
        writer.writeheader(); writer.writerows(manifest_rows)
    return manifest
