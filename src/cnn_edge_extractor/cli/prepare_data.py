import argparse
from pathlib import Path

from src.cnn_edge_extractor.data.prepare import raster_to_binary_mask, crop_aligned_tiles, split_tiles
from src.cnn_edge_extractor.utils.config import read_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare data: convert labels to binary mask, crop tiles, split train/val/test.')
    parser.add_argument('--config', type=Path, default=Path('configs/data.yaml'))
    args = parser.parse_args(); cfg = read_simple_yaml(args.config)
    dem_path = Path(cfg['raw_dem']); label_path = Path(cfg['raw_labels_dxf']); processed_dir = Path(cfg['processed_dir'])
    tile_size = int(cfg.get('tile_size', 100)); stride = int(cfg.get('stride', tile_size))
    train_ratio = float(cfg.get('train_ratio', 0.6)); val_ratio = float(cfg.get('val_ratio', 0.2)); seed = int(cfg.get('seed', 42))
    mask_path = processed_dir / 'full_mask.tif'; raster_to_binary_mask(label_path, mask_path)
    records = crop_aligned_tiles(dem_path, mask_path, processed_dir, tile_size=tile_size, stride=stride)
    manifest = split_tiles(records, processed_dir, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    print(f'Prepared {len(records)} tiles'); print(manifest)

if __name__ == '__main__':
    main()
