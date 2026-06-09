import torch
import torch.optim as optim

from src.cnn_edge_extractor.models.unet import UNet
from src.cnn_edge_extractor.training.losses import CombinedLoss


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device: torch.device):
    return UNet(in_channels=1, out_channels=1).to(device)


def build_optimizer(model, lr: float):
    return optim.Adam(model.parameters(), lr=lr)


def build_criterion():
    return CombinedLoss()
