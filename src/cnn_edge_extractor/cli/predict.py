import argparse
from pathlib import Path

from src.cnn_edge_extractor.inference.predictor import DEMPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description='Run sliding-window inference on a full DEM.')
    parser.add_argument('--checkpoint', type=Path, required=False)
    parser.add_argument('--dem', type=Path, required=False)
    parser.add_argument('--out', type=Path, required=False)
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    if not (args.checkpoint and args.dem and args.out):
        print('Provide --checkpoint, --dem and --out for actual prediction.')
        return
    predictor = DEMPredictor(args.checkpoint, device=args.device, window_size=args.window_size, overlap=args.overlap)
    mask_path, prob_path = predictor.predict_full_dem(args.dem, args.out, threshold=args.threshold)
    print(mask_path)
    print(prob_path)


if __name__ == '__main__':
    main()
