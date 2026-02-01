# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from QuarryEdgeDataset import create_dataloaders


# ============================================================================
# U-NET АРХИТЕКТУРА (ИСПРАВЛЕНО)
# ============================================================================

class DoubleConv(nn.Module):
    """Двойная свёртка"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net с автоматическим выравниванием размеров"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder с выравниванием размеров
        dec3 = self.upconv3(bottleneck)
        dec3 = self._crop_and_concat(dec3, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._crop_and_concat(dec2, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._crop_and_concat(dec1, enc1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out(dec1))

    def _crop_and_concat(self, upsampled, encoder_features):
        """
        Выравнивает размеры и конкатенирует тензоры.

        Args:
            upsampled: тензор после upsampling
            encoder_features: тензор из encoder (skip connection)
        """
        # Если размеры не совпадают, обрезаем или дополняем
        if upsampled.shape != encoder_features.shape:
            # Вычисляем разницу
            diff_h = encoder_features.size(2) - upsampled.size(2)
            diff_w = encoder_features.size(3) - upsampled.size(3)

            # Дополняем upsampled если он меньше
            if diff_h > 0 or diff_w > 0:
                upsampled = F.pad(upsampled, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2
                ])
            # Обрезаем upsampled если он больше
            elif diff_h < 0 or diff_w < 0:
                upsampled = upsampled[:, :,
                :encoder_features.size(2),
                :encoder_features.size(3)]

        return torch.cat([upsampled, encoder_features], dim=1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss для сегментации линий"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice Loss"""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice(pred, target)


# ============================================================================
# МЕТРИКИ
# ============================================================================

def calculate_iou(pred, target, threshold=0.5):
    """Intersection over Union"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_f1(pred, target, threshold=0.5):
    """F1-score"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.item()


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for dem, mask in pbar:
        dem = dem.to(device)
        mask = mask.to(device)

        # Forward
        optimizer.zero_grad()
        output = model(dem)
        loss = criterion(output, mask)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_iou += calculate_iou(output.detach(), mask.detach())
        running_f1 += calculate_f1(output.detach(), mask.detach())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{calculate_iou(output.detach(), mask.detach()):.4f}'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_f1 = running_f1 / len(dataloader)

    return epoch_loss, epoch_iou, epoch_f1


def validate(model, dataloader, criterion, device):
    """Валидация"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for dem, mask in pbar:
            dem = dem.to(device)
            mask = mask.to(device)

            output = model(dem)
            loss = criterion(output, mask)

            running_loss += loss.item()
            running_iou += calculate_iou(output, mask)
            running_f1 += calculate_f1(output, mask)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{calculate_iou(output, mask):.4f}'
            })

    val_loss = running_loss / len(dataloader)
    val_iou = running_iou / len(dataloader)
    val_f1 = running_f1 / len(dataloader)

    return val_loss, val_iou, val_f1


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # IoU
    axes[1].plot(history['train_iou'], label='Train IoU')
    axes[1].plot(history['val_iou'], label='Val IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU')
    axes[1].legend()
    axes[1].grid(True)

    # F1
    axes[2].plot(history['train_f1'], label='Train F1')
    axes[2].plot(history['val_f1'], label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-score')
    axes[2].set_title('F1-score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ График сохранён: {save_path}")
    plt.show()


def visualize_predictions(model, dataloader, device, num_samples=1, save_path='predictions.png'):
    """Визуализация предсказаний"""
    model.eval()

    samples = []
    with torch.no_grad():
        for dem, mask in dataloader:
            if len(samples) >= num_samples:
                break

            dem = dem.to(device)
            output = model(dem)

            samples.append({
                'dem': dem.cpu(),
                'mask': mask,
                'pred': output.cpu()
            })

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, sample in enumerate(samples):
        dem_np = sample['dem'][0].squeeze().numpy()
        mask_np = sample['mask'][0].squeeze().numpy()
        pred_np = sample['pred'][0].squeeze().numpy()

        axes[i, 0].imshow(dem_np, cmap='terrain')
        axes[i, 0].set_title('Input DEM')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(dem_np, cmap='terrain')
        axes[i, 3].imshow(pred_np, cmap='Reds', alpha=0.5)
        axes[i, 3].set_title('DEM + Prediction')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Предсказания: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "1m" / "100"
    MODEL_SAVE_PATH = Path("models")
    MODEL_SAVE_PATH.mkdir(exist_ok=True)

    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 70)
    print("EDGE EXTRACTOR — ОБУЧЕНИЕ U-NET")
    print("=" * 70)
    print(f"Устройство: {DEVICE}")
    print(f"Данные: {DATA_DIR.absolute()}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=0.6,
        val_ratio=0.2,
        normalize_type="zscore"
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = CombinedLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    history = {
        'train_loss': [], 'train_iou': [], 'train_f1': [],
        'val_loss': [], 'val_iou': [], 'val_f1': []
    }

    best_val_iou = 0.0

    print("\n" + "=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 70 + "\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        train_loss, train_iou, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)

        print(f"\nРезультаты эпохи {epoch + 1}:")
        print(f"  Train — Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_f1': val_f1
            }, MODEL_SAVE_PATH / 'best_edge_extractor.pth')
            print(f"  ✓ Модель сохранена (IoU: {val_iou:.4f})")

    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"Лучший Val IoU: {best_val_iou:.4f}")

    plot_training_history(history, save_path='training_history.png')
    print("\nГенерация предсказаний...")
    visualize_predictions(model, val_loader, DEVICE, num_samples=1, save_path='predictions.png')

    print(f"\n✓ Модель: {MODEL_SAVE_PATH / 'best_edge_extractor.pth'}")
    print("✓ Готово!")


if __name__ == "__main__":
    main()
