import argparse
from pathlib import Path

from src.cnn_edge_extractor.training.engine import Trainer
from src.cnn_edge_extractor.utils.config import read_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-Net on processed DEM tiles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to training config file",
    )
    args = parser.parse_args()

    cfg = read_simple_yaml(args.config)

    trainer = Trainer(
        data_dir=Path(cfg["data_dir"]),
        checkpoint_dir=Path(cfg["checkpoint_dir"]),
        batch_size=int(cfg.get("batch_size", 4)),
        epochs=int(cfg.get("epochs", 50)),
        learning_rate=float(cfg.get("learning_rate", 0.001)),
        num_workers=int(cfg.get("num_workers", 0)),
        normalize_type=str(cfg.get("normalize_type", "zscore")),
        device=str(cfg.get("device", "auto")),
        threshold=float(cfg.get("threshold", 0.5)),
    )

    history, metrics = trainer.fit()

    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Epochs: {len(history)}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test IoU : {metrics['test_iou']:.4f}")
    print(f"Test F1  : {metrics['test_f1']:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()