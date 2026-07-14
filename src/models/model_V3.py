# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import time
import os
import yaml
import argparse
from pathlib import Path
from dataset import NMRDataset
from model import NMR_Net

class WeightedConstrainedMSELoss(nn.Module):
    def __init__(self, lambda_sum=0.5, device='cuda'):
        super(WeightedConstrainedMSELoss, self).__init__()
        self.lambda_sum = lambda_sum
        
        # Orden: ["CH3", "CH2", "CH", "Cq", "CH3-X", "CH2-X", "CH-X", "Cq-X", "=CH2", "=CH/Ar", "=Cq/Ar", "Aldeh", "C=O"]
        # PESO ALTO (10.0) a los grupos fáciles/visibles, PESO BAJO (1.0) a los difíciles/invisibles
        weights = [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 1.0, 1.0, 10.0, 1.0]
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def forward(self, pred, target):
        # MSE ponderado
        loss_individual = torch.mean(self.weights * (pred - target)**2)
        # Suma total restringida
        loss_sum = torch.mean((torch.sum(pred, dim=1) - torch.sum(target, dim=1))**2)
        return loss_individual + (self.lambda_sum * loss_sum)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train(config_path):
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO V1 PONDERADO: {cfg['experiment_name']} ---")
    
    base_dir = Path(cfg['paths']['base_dir'])
    h5_path = base_dir / cfg['paths']['h5_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    ckpt_dir = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRDataset(str(h5_path), str(labels_path))
    val_size = int(len(full_dataset) * cfg['hyperparameters']['val_split'])
    train_size = len(full_dataset) - val_size
    
    # --- FIJAMOS LA SEMILLA AQUI ---
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    use_pin_memory = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'], 
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'], 
                            shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin_memory)

    model = NMR_Net(num_classes=13).to(device)
    
    # Pasamos el device a la funcion de perdida
    criterion = WeightedConstrainedMSELoss(lambda_sum=0.5, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = cfg['hyperparameters']['epochs']
    print(f"\n[START] Comenzando entrenamiento ({epochs} epochs)...")
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            hsqc_img = inputs[0].to(device)
            proj_1d = inputs[1].to(device)
            total_sig = inputs[2].to(device)
            
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(hsqc_img, proj_1d, total_sig)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        val_loss = validate(model, val_loader, criterion, device)
        avg_train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[EPOCH {epoch+1}] Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / f"{cfg['experiment_name']}_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] Nuevo mejor modelo guardado!")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_ep{epoch+1}.pth")

    print(f"\n[DONE] Finalizado en {(time.time() - start_time) / 60:.1f} min. Mejor Val: {best_val_loss:.4f}")

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            hsqc_img = inputs[0].to(device)
            proj_1d = inputs[1].to(device)
            total_sig = inputs[2].to(device)
            targets = targets.to(device)
            
            outputs = model(hsqc_img, proj_1d, total_sig)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)