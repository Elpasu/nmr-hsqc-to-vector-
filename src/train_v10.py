# coding: ascii
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path
from dataset_v10 import NMRDataset
from model_v10 import NMR_Net

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

def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO V10 (2CH + FM + 19v): {cfg['experiment_name']} ---")

    base_dir    = Path(cfg['paths']['base_dir'])
    h5_path     = base_dir / cfg['paths']['h5_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir    = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
    val_size   = int(len(full_dataset) * cfg['hyperparameters']['val_split'])
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    model     = NMR_Net(num_classes=19).to(device)
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
