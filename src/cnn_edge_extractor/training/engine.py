from __future__ import annotations
from pathlib import Path
import csv, json
import torch
import torch.optim as optim
from tqdm import tqdm

from src.cnn_edge_extractor.data.dataset import create_dataloaders
from src.cnn_edge_extractor.models.unet import UNet
from src.cnn_edge_extractor.training.losses import CombinedLoss
from src.cnn_edge_extractor.training.metrics import calculate_iou, calculate_f1


class Trainer:
    def __init__(self, data_dir: Path, checkpoint_dir: Path, batch_size: int = 4, epochs: int = 50, learning_rate: float = 1e-3, num_workers: int = 0, normalize_type: str = 'zscore', device: str = 'auto', threshold: float = 0.5):
        self.data_dir = data_dir; self.checkpoint_dir = checkpoint_dir; self.batch_size = batch_size; self.epochs = epochs
        self.learning_rate = learning_rate; self.num_workers = num_workers; self.normalize_type = normalize_type; self.threshold = threshold
        if device == 'auto':
            if torch.cuda.is_available(): self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = torch.device('mps')
            else: self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.criterion = CombinedLoss(); self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def _run_epoch(self, dataloader, train: bool = True):
        self.model.train(train); running_loss = running_iou = running_f1 = 0.0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            pbar = tqdm(dataloader, desc='Train' if train else 'Val')
            for dem, mask in pbar:
                dem = dem.to(self.device); mask = mask.to(self.device)
                if train: self.optimizer.zero_grad()
                output = self.model(dem); loss = self.criterion(output, mask)
                if train: loss.backward(); self.optimizer.step()
                batch_iou = calculate_iou(output.detach(), mask.detach(), self.threshold)
                batch_f1 = calculate_f1(output.detach(), mask.detach(), self.threshold)
                running_loss += loss.item(); running_iou += batch_iou; running_f1 += batch_f1
                pbar.set_postfix(loss=f'{loss.item():.4f}', iou=f'{batch_iou:.4f}', f1=f'{batch_f1:.4f}')
        n = max(len(dataloader), 1)
        return running_loss / n, running_iou / n, running_f1 / n
    def fit(self):
        train_loader, val_loader, test_loader = create_dataloaders(self.data_dir, batch_size=self.batch_size, num_workers=self.num_workers, normalize_type=self.normalize_type)
        history = []; best_iou = -1.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_iou, train_f1 = self._run_epoch(train_loader, train=True)
            val_loss, val_iou, val_f1 = self._run_epoch(val_loader, train=False)
            row = {'epoch': epoch, 'train_loss': train_loss, 'train_iou': train_iou, 'train_f1': train_f1, 'val_loss': val_loss, 'val_iou': val_iou, 'val_f1': val_f1}
            history.append(row); self._save_checkpoint(self.checkpoint_dir / 'last.ckpt', epoch, val_iou, val_f1)
            if val_iou > best_iou: best_iou = val_iou; self._save_checkpoint(self.checkpoint_dir / 'best.ckpt', epoch, val_iou, val_f1)
        self._write_history(history)
        test_loss, test_iou, test_f1 = self._run_epoch(test_loader, train=False)
        metrics = {'test_loss': test_loss, 'test_iou': test_iou, 'test_f1': test_f1}
        (self.checkpoint_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        return history, metrics
    def _save_checkpoint(self, path: Path, epoch: int, val_iou: float, val_f1: float):
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'val_iou': val_iou, 'val_f1': val_f1}, path)
    def _write_history(self, history: list[dict]):
        csv_path = self.checkpoint_dir / 'history.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys())); writer.writeheader(); writer.writerows(history)
