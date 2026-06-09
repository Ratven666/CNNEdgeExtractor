from __future__ import annotations
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt
import torch

from src.cnn_edge_extractor.data.dataset import create_dataloaders
from src.cnn_edge_extractor.models.unet import UNet
from src.cnn_edge_extractor.training.losses import CombinedLoss
from src.cnn_edge_extractor.training.metrics import calculate_iou, calculate_f1


def evaluate_checkpoint(data_dir: Path, checkpoint_path: Path, output_dir: Path, normalize_type: str = 'zscore', num_workers: int = 0, threshold: float = 0.5, device: str = 'auto'):
    if device == 'auto':
        if torch.cuda.is_available(): dev = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): dev = torch.device('mps')
        else: dev = torch.device('cpu')
    else:
        dev = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = create_dataloaders(data_dir, batch_size=1, num_workers=num_workers, normalize_type=normalize_type)
    model = UNet(in_channels=1, out_channels=1).to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval(); criterion = CombinedLoss(); rows = []
    with torch.no_grad():
        for idx, (dem, mask) in enumerate(test_loader):
            dem = dem.to(dev); mask = mask.to(dev); pred = model(dem)
            loss = criterion(pred, mask).item(); iou = calculate_iou(pred, mask, threshold); f1 = calculate_f1(pred, mask, threshold)
            rows.append({'sample': idx, 'loss': loss, 'iou': iou, 'f1': f1})
            if idx < 3:
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(dem.cpu().squeeze().numpy(), cmap='terrain'); axes[0].set_title('DEM')
                axes[1].imshow(mask.cpu().squeeze().numpy(), cmap='gray'); axes[1].set_title('Mask')
                axes[2].imshow((pred.cpu().squeeze().numpy() > threshold).astype('uint8'), cmap='gray'); axes[2].set_title('Pred')
                [ax.axis('off') for ax in axes]; fig.tight_layout(); fig.savefig(output_dir / f'sample_{idx:02d}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    csv_path = output_dir / 'metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample', 'loss', 'iou', 'f1']); writer.writeheader(); writer.writerows(rows)
    summary = {'mean_loss': sum(r['loss'] for r in rows) / len(rows) if rows else None, 'mean_iou': sum(r['iou'] for r in rows) / len(rows) if rows else None, 'mean_f1': sum(r['f1'] for r in rows) / len(rows) if rows else None, 'samples': len(rows)}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
