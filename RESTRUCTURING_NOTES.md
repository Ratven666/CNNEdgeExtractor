# Restructuring notes

## What was changed

- Introduced `src/cnn_edge_extractor/` package layout.
- Split future responsibilities into `data/`, `models/`, `training/`, `inference/`, `cli/`, `utils/`.
- Copied original ML scripts into `*_legacy.py` files so nothing is lost during migration.
- Added `pyproject.toml` and console entrypoints.
- Added YAML configs for data, training, and prediction.
- Copied repository sample data into `data/raw/`.

## Mapping from original files

- `app/cnn_pytorch/DEMPredictor.py` -> `src/cnn_edge_extractor/inference/predictor_legacy.py`
- `app/cnn_pytorch/DoubleConv.py` -> `src/cnn_edge_extractor/training/train_legacy.py`
- `app/cnn_pytorch/train_unet.py` -> `src/cnn_edge_extractor/training/train_unet_legacy.py`
- `app/cnn_pytorch/QuarryEdgeDataset.py` -> `src/cnn_edge_extractor/data/dataset_legacy.py`
- `app/cnn_pytorch/aug_unet_learning.py` -> `src/cnn_edge_extractor/data/augment_legacy.py`
- `utils/data_cropper.py` -> `src/cnn_edge_extractor/data/tiling_legacy.py`
- `utils/dxf_lines_to_geotiff.py` -> `src/cnn_edge_extractor/data/rasterize_labels_legacy.py`

## Immediate next steps

1. Port training loop from legacy script into `training/engine.py`.
2. Port visualization helpers into `visualization/plots.py`.
3. Implement `prepare_data.py` using rasterization + tiling + deterministic split manifest.
4. Add tests and one end-to-end smoke example.

## Added in second pass

- Implemented `data/prepare.py` for binary mask conversion, aligned tiling, and train/val/test split manifest generation.
- Implemented `training/engine.py` with checkpoint saving and history CSV.
- Implemented `training/evaluate.py` with sample visualizations and summary metrics.
- Updated CLI commands so `prepare_data`, `train`, and `evaluate` are now wired to concrete modules.
- The sandbox used here does not currently provide all runtime dependencies (for example `rasterio`), so end-to-end execution could not be validated inside this environment.
