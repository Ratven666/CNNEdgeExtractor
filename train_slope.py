from __future__ import annotations

from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cnn_edge_extractor.data.dataset import create_dataloaders
from src.cnn_edge_extractor.models.unet import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_iou_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    f1 = (2 * intersection + eps) / (preds.sum() + targets.sum() + eps)

    return iou.item(), f1.item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n_batches = 0

    progress = tqdm(loader, desc=desc, leave=False)

    for images, masks in progress:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

        iou, f1 = compute_iou_f1(logits.detach(), masks.detach())

        total_loss += loss.item()
        total_iou += iou
        total_f1 += f1
        n_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou:.4f}",
            f1=f"{f1:.4f}",
        )

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    return total_loss / n_batches, total_iou / n_batches, total_f1 / n_batches


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data" / "processed_slope"
    checkpoints_dir = project_root / "models" / "checkpoints_slope"

    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 50
    num_workers = 0
    normalize_type = "zscore"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("TRAINING ON SLOPE DATASET")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Checkpoints dir: {checkpoints_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_type=normalize_type,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss, train_iou, train_f1 = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc="Train",
        )

        val_loss, val_iou, val_f1 = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            desc="Val",
        )

        print(f"Train | loss={train_loss:.4f} iou={train_iou:.4f} f1={train_f1:.4f}")
        print(f"Val   | loss={val_loss:.4f} iou={val_iou:.4f} f1={val_f1:.4f}")

        metrics = {
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": val_f1,
        }

        save_checkpoint(
            path=checkpoints_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

            save_checkpoint(
                path=checkpoints_dir / "best.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )

            print(f"New best checkpoint saved. val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_iou, test_f1 = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        desc="Test",
    )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test IoU : {test_iou:.4f}")
    print(f"Test F1  : {test_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()