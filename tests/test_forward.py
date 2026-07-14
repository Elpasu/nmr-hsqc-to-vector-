# coding: ascii
import torch
from torch.utils.data import DataLoader
from dataset_v10 import NMRDataset
from model_v10 import NMR_Net

D = "/home/lpassaglia.iquir/DB_200k"
print("Cargando dataset (calcula FM, tarda ~1 min)...")
ds = NMRDataset(f"{D}/nmr_dataset_v3_202465.h5",
                f"{D}/vectors_13c_19v_202465.npy",
                f"{D}/smiles_202465.npy")
print(f"N moleculas: {len(ds)}")

loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
(hsqc, proj, cond), target = next(iter(loader))
print(f"hsqc  : {hsqc.shape}   (esperado [4, 2, 256, 256])")
print(f"proj  : {proj.shape}   (esperado [4, 512])")
print(f"cond  : {cond.shape}   (esperado [4, 8])")
print(f"target: {target.shape} (esperado [4, 19])")

model = NMR_Net(num_classes=19)
out = model(hsqc, proj, cond)
print(f"OUTPUT: {out.shape}    (esperado [4, 19])")
print("\n>>> FORWARD OK - listo para entrenar <<<")
