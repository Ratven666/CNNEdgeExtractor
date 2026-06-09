# CNNEdgeExtractor (restructured)

This is a cleaned project layout for quarry edge extraction from DEM rasters.

## Pipeline

1. `cnne-prepare-data` — rasterize DXF labels, crop DEM/masks to tiles, split train/val/test.
2. `cnne-train` — train U-Net on processed tiles and save checkpoints.
3. `cnne-predict` — run sliding-window inference on a full DEM.
4. `cnne-evaluate` — compute metrics and create visual outputs.

## Why this layout

The original repository mixes domain logic, dataset preparation, training, and prediction in a few overlapping modules. This structure separates:

- `domain/` — scan, mesh, DEM, morphometry.
- `data/` — label rasterization, tiling, dataset loading, augmentation, manifests.
- `models/` — model architectures only.
- `training/` — losses, metrics, loops, checkpoints.
- `inference/` — predictors and sliding-window logic.
- `cli/` — explicit user-facing entrypoints.

## Current sample data included

Raw sample files from the original repository were copied into `data/raw/`, including:

- `data/raw/1m/grib_1m.tif`
- `data/raw/Grib_dxf_mesh.dxf`
- tile pairs under `data/raw/1m/100/`

## Recommended next refactor

- Move reusable code from `*_legacy.py` into the new modules incrementally.
- Extract `UNet` from legacy training scripts into `models/unet.py`.
- Replace hardcoded paths with YAML configs.
- Add tests for dataset loading, forward pass, and sliding-window prediction.
