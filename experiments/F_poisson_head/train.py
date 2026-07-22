# experiments/F_poisson_head/train.py
# coding: ascii
"""
train.py -- Exp F: cabeza Poisson + entrenamiento extendido sobre el Set
Transformer de Exp E Fase 3. Cambios respecto a Fase 3:
  (1) modelo con activacion softplus en la salida (model_f_settransformer.py) --
      garantiza lambda >= 0, que exige PoissonNLLLoss(log_input=False).
  (2) ConstrainedPoissonLoss en vez de ConstrainedMSELoss (Poisson NLL por
      clase + mismo termino de restriccion de suma, lambda_sum=0.5).
  (3) epochs=250 en vez de 100 (en Fase 3 el LR nunca bajo de 0.001 en 100
      epocas -- scheduler patience=8/factor=0.7 sin cambios, regla 6).
Dataset, split congelado (Exp D), condicionante FM y arquitectura del Set
Transformer (d_model/n_heads/n_layers/n_seeds) identicos a Fase 3.

CAVEAT: el val loss reportado es Poisson NLL, NO es comparable numericamente
contra el 0.0097 (MSE) de Fase 3. Comparar por EMA (evaluate.py).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_f import NMRTwoSetsDataset
from model_f_settransformer import NMR_SetTransformer
from split_utils import canonicalize_smiles, remove_leaking_from_train


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConstrainedPoissonLoss(nn.Module):
    """Poisson NLL por clase (el modelo ya devuelve lambda >= 0 via
    softplus, por eso log_input=False) + el mismo termino de restriccion de
    suma que ConstrainedMSELoss (MSE sobre el total predicho vs el total
    real, lambda_sum=0.5). full=True agrega el termino de Stirling
    (aproximacion de log(target!)) para que la loss sea una NLL propiamente
    normalizada, no solo la parte que importa para el gradiente."""
    def __init__(self, lambda_sum=0.5):
        super().__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=False, full=True)
        self.mse = nn.MSELoss()
        self.lambda_sum = lambda_sum

    def forward(self, pred, target):
        li = self.poisson(pred, target)
        ls = self.mse(torch.sum(pred, dim=1), torch.sum(target, dim=1))
        return li + self.lambda_sum * ls


def load_config(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(cfg, num_classes=19):
    m = cfg['model']
    return NMR_SetTransformer(
        num_classes=num_classes,
        d_model=int(m.get('d_model', 64)),
        n_heads=int(m.get('n_heads', 4)),
        n_layers=int(m.get('n_layers', 2)),
        n_seeds=int(m.get('n_seeds', 1)),
    )


def build_frozen_split(full_dataset, base_dir, cfg):
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


def unpack(inputs, device):
    return (inputs[0].to(device), inputs[1].to(device),
            inputs[2].to(device), inputs[3].to(device), inputs[4].to(device))


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            pch, mch, p13, m13, cond = unpack(inputs, device)
            targets = targets.to(device)
            total += criterion(model(pch, mch, p13, m13, cond), targets).item()
    return total / len(loader)


def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP F (Poisson + entrenamiento extendido): {cfg['experiment_name']} ---")

    base_dir = Path(cfg['paths']['base_dir'])
    peaks_ch = base_dir / cfg['paths']['peaks_ch_filename']
    peaks_13c = base_dir / cfg['paths']['peaks_13c_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                                     str(smiles_path), cfg['normalization'])
    train_idx, val_idx = build_frozen_split(full_dataset, base_dir, cfg)
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist())

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                            shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    model = build_model(cfg, num_classes=19).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametros totales del modelo: {n_params:,} "
          f"(igual a Fase 3 Set Transformer, ~70k; softplus no agrega parametros)")

    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    sched_cfg = cfg['hyperparameters'].get('scheduler', {})
    patience = sched_cfg.get('patience', 8); factor = sched_cfg.get('factor', 0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    print(f"[INFO] Scheduler: patience={patience}, factor={factor}")
    print("[INFO] Loss: ConstrainedPoissonLoss -- el val loss NO es comparable "
          "numericamente contra el 0.0097 (MSE) de Fase 3. Comparar por EMA.")

    epochs = cfg['hyperparameters']['epochs']
    print(f"\n[START] {epochs} epochs...")
    start_time = time.time(); best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train(); running_loss = 0.0; epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            pch, mch, p13, m13, cond = unpack(inputs, device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(pch, mch, p13, m13, cond)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_best.pth")
            print("[SAVE] Nuevo mejor modelo!")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_ep{epoch+1}.pth")

    print(f"\n[DONE] {(time.time()-start_time)/60:.1f} min. Mejor Val: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
