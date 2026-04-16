"""
Quick evaluation script: loads the best checkpoint and prints per-action accuracy.
Run from 2_week_simulator/:
    uv run python -m curved_road.exp1.eval
"""
from __future__ import annotations
from pathlib import Path
import torch
from shared.utils.config import load_config, DataConfig
from shared.data.dataset import load_datasets, make_dataloaders
from shared.data.transforms import get_val_transform
from shared.training.metrics import compute_metrics
from curved_road.exp1.model import Exp1Model

def main() -> None:
    cfg = load_config(Path(__file__).parent / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        training_data_dir=str(Path(__file__).parent.parent.parent / cfg.data.training_data_dir),
        val_split=cfg.data.val_split,
        seed=cfg.data.seed,
    )
    _, val_dataset = load_datasets(cfg=data_cfg, train_transform=get_val_transform(), val_transform=get_val_transform())
    _, val_loader = make_dataloaders(val_dataset, val_dataset, cfg.training)

    model = Exp1Model(cfg.model.feature_dim, cfg.model.mlp_hidden_dim, cfg.model.num_actions).to(device)
    ckpt = Path(__file__).parent.parent.parent / cfg.results_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            all_logits.append(model(batch[0].to(device)))
            all_labels.append(batch[1].to(device))

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    print(f"Aggregate:  {metrics['acc']:.1%}")
    for action in ["forward", "backward", "left", "right"]:
        print(f"  {action:<10} {metrics[f'acc_{action}']:.1%}")

if __name__ == "__main__":
    main()
