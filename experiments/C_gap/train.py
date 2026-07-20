# coding: ascii
"""
train.py -- Exp C: rebalanceo de la fusion (GAP) sobre el baseline V10.
Sin regularizacion (dropout/weight_decay abandonados tras la falla de
Exp B: dieron underfitting real). Usa el split congelado de Exp D
(val_indices_frozen.npy); el train set se reconstruye con la misma
logica de dedup/leak que uso split.py originalmente (copiada a
split_utils.py), sin regenerar el archivo ni depender de otras carpetas
de experimento.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_v10 import NMRDataset
from model_c import NMR_Net
from split_utils import canonicalize_smiles, remove_leaking_from_train


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConstrainedMSELoss(nn.Module):
    def __init__(self, lambda_sum=0.5):
        super().__init__()
        self.mse = nn.MSELoss(); self.lambda_sum = lambda_sum

    def forward(self, pred, target):
        li = self.mse(pred, target)
        ls = self.mse(torch.sum(pred, dim=1), torch.sum(target, dim=1))
        return li + self.lambda_sum * ls


def load_config(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_frozen_split(full_dataset, base_dir, cfg):
    """Reconstruye train/val a partir de val_indices_frozen.npy (Exp D).
    val queda EXACTAMENTE como esta en el archivo (nunca se toca). train
    es todo lo demas, menos las filas que comparten SMILES canonico con
    alguna de val (misma logica de leak que uso split.py de Exp D)."""
    val_indices_path = base_dir / cfg['paths']['val_indices_filename']
    if not os.path.exists(val_indices_path):
        raise FileNotFoundError(
            f"No se encontro el split congelado en: {val_indices_path}\n"
            "Corri primero experiments/D_val_congelado/split.py (Exp D)."
        )
    val_idx = np.load(val_indices_path)

    smiles_path = base_dir / cfg['paths']['smiles_filename']
    smiles = np.load(smiles_path, allow_pickle=True)
    canonical, n_invalid = canonicalize_smiles(smiles)

    all_idx = np.arange(len(full_dataset))
    train_idx_raw = np.setdiff1d(all_idx, val_idx, assume_unique=False)
    train_idx, n_removed = remove_leaking_from_train(train_idx_raw, val_idx, canonical)

    print(f"[INFO] Split congelado: SMILES invalidos={n_invalid} | "
          f"train={len(train_idx)} (leak removido={n_removed}) | val={len(val_idx)}")
    return train_idx, val_idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP C (V10 + GAP, sin regularizacion): {cfg['experiment_name']} ---")

    base_dir    = Path(cfg['paths']['base_dir'])
    h5_path     = base_dir / cfg['paths']['h5_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir    = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
    train_idx, val_idx = build_frozen_split(full_dataset, base_dir, cfg)
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds   = Subset(full_dataset, val_idx.tolist())

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    model    = NMR_Net(num_classes=19).to(device)
    n_params = count_params(model)
    print(f"[INFO] Parametros totales del modelo: {n_params:,} (V10 original: ~8,603,299)")

    criterion = ConstrainedMSELoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])

    sched_cfg = cfg['hyperparameters'].get('scheduler', {})
    patience  = sched_cfg.get('patience', 8)
    factor    = sched_cfg.get('factor', 0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    print(f"[INFO] Scheduler: patience={patience}, factor={factor}")

    epochs = cfg['hyperparameters']['epochs']
    print(f"\n[START] {epochs} epochs...")
    start_time = time.time(); best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train(); running_loss = 0.0; epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            hsqc = inputs[0].to(device); proj = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(hsqc, proj, cond)
            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        if device.type == 'cuda':
            torch.cuda.synchronize()
        val_loss = validate(model, val_loader, criterion, device)
        avg_train = running_loss / len(train_loader)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"[EPOCH {epoch+1}] Train: {avg_train:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f} | Time: {time.time()-epoch_start:.1f}s")

        if (epoch + 1) in (10, 20, 30) and val_loss > 0.10:
            print(f"[WARN] Val loss ({val_loss:.4f}) todavia muy por encima de la referencia "
                  f"de V10 (0.031) en la epoca {epoch+1}. Si sigue asi hasta ~ep30, es la misma "
                  f"senal de underfitting que tumbo a Exp B -- ver RATIONALE.md antes de esperar "
                  f"a que termine el entrenamiento completo.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_best.pth")
            print("[SAVE] Nuevo mejor modelo!")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_ep{epoch+1}.pth")

    print(f"\n[DONE] {(time.time()-start_time)/60:.1f} min. Mejor Val: {best_val_loss:.4f}")


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            hsqc = inputs[0].to(device); proj = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            total += criterion(model(hsqc, proj, cond), targets).item()
    return total / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
