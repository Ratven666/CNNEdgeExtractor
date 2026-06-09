import argparse
from pathlib import Path

from src.cnn_edge_extractor.training.evaluate import evaluate_checkpoint
from src.cnn_edge_extractor.utils.config import read_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate model predictions on the test split.')
    parser.add_argument('--config', type=Path, default=Path('configs/train.yaml'))
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=Path('outputs/eval'))
    args = parser.parse_args(); cfg = read_simple_yaml(args.config)
    checkpoint = args.checkpoint or Path(cfg['checkpoint_dir']) / 'best.ckpt'
    summary = evaluate_checkpoint(data_dir=Path(cfg['data_dir']), checkpoint_path=checkpoint, output_dir=args.out, normalize_type=str(cfg.get('normalize_type', 'zscore')), num_workers=int(cfg.get('num_workers', 0)), threshold=float(cfg.get('threshold', 0.5)), device=str(cfg.get('device', 'auto')))
    print(summary)

if __name__ == '__main__':
    main()
