from src.cnn_edge_extractor.configs import TrainConfig
from src.cnn_edge_extractor.data.dataset import create_dataloaders
from src.cnn_edge_extractor.training.engine import Trainer
from src.cnn_edge_extractor.training.factory import resolve_device, build_model, build_optimizer, build_criterion


def run_train(cfg: TrainConfig):
    device = resolve_device(cfg.device)
    model = build_model(device)
    optimizer = build_optimizer(model, cfg.learning_rate)
    criterion = build_criterion()

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        normalize_type=cfg.normalize_type,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        epochs=cfg.epochs,
        threshold=cfg.threshold,
    )
    return trainer.fit(train_loader, val_loader, test_loader)
