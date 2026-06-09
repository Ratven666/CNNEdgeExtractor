from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path
    checkpoint_dir: Path
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 1e-3
    num_workers: int = 0
    normalize_type: str = "zscore"
    device: str = "auto"
    threshold: float = 0.5