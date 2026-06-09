import argparse
from pathlib import Path

from src.cnn_edge_extractor.configs import TrainConfig
from src.cnn_edge_extractor.training.run import run_train
from src.cnn_edge_extractor.utils.config import read_simple_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()

    raw = read_simple_yaml(args.config)
    cfg = TrainConfig(
        data_dir=Path(raw["data_dir"]),
        checkpoint_dir=Path(raw["checkpoint_dir"]),
        batch_size=int(raw.get("batch_size", 4)),
        epochs=int(raw.get("epochs", 50)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        num_workers=int(raw.get("num_workers", 0)),
        normalize_type=str(raw.get("normalize_type", "zscore")),
        device=str(raw.get("device", "auto")),
        threshold=float(raw.get("threshold", 0.5)),
    )

    run_train(cfg)


if __name__ == "__main__":
    main()