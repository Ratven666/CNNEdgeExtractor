from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

from src.cnn_edge_extractor.models.unet import UNet


class DEMPredictor:
    def __init__(self, model_path: Path, device: str = 'auto', window_size: int = 100, overlap: int = 20):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def normalize_dem(dem: np.ndarray) -> np.ndarray:
        mean = np.mean(dem)
        std = np.std(dem)
        if std > 1e-6:
            dem = (dem - mean) / std
        return dem

    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        tile = self.normalize_dem(tile)
        tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tile_tensor)
        return pred.cpu().squeeze().numpy()

    @staticmethod
    def _create_weight_mask(size: int) -> np.ndarray:
        center = size // 2
        y, x = np.ogrid[:size, :size]
        sigma = size / 4
        weight = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        return weight / weight.max()

    def predict_full_dem(self, dem_path: Path, output_path: Path, threshold: float = 0.5):
        with rasterio.open(dem_path) as src:
            height, width = src.height, src.width
            prediction_full = np.zeros((height, width), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)
            n_rows = (height - self.window_size) // self.stride + 1
            n_cols = (width - self.window_size) // self.stride + 1
            weight = self._create_weight_mask(self.window_size)
            for row_idx in tqdm(range(n_rows), desc='Predict rows'):
                for col_idx in range(n_cols):
                    row_start = row_idx * self.stride
                    col_start = col_idx * self.stride
                    window = Window(col_start, row_start, self.window_size, self.window_size)
                    tile = src.read(1, window=window).astype(np.float32)
                    tile = np.nan_to_num(tile, nan=0.0)
                    pred_tile = self.predict_tile(tile)
                    prediction_full[row_start:row_start+self.window_size, col_start:col_start+self.window_size] += pred_tile * weight
                    weight_map[row_start:row_start+self.window_size, col_start:col_start+self.window_size] += weight
            prediction_full = np.divide(prediction_full, weight_map, out=np.zeros_like(prediction_full), where=weight_map != 0)
            prediction_binary = (prediction_full > threshold).astype(np.uint8)
            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=None)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(prediction_binary, 1)
            prob_path = Path(str(output_path).replace('.tif', '_prob.tif'))
            profile_prob = src.profile.copy()
            profile_prob.update(dtype=rasterio.float32, count=1, compress='lzw')
            with rasterio.open(prob_path, 'w', **profile_prob) as dst:
                dst.write(prediction_full.astype(np.float32), 1)
            return output_path, prob_path
