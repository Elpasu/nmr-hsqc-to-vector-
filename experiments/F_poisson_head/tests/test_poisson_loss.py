# experiments/F_poisson_head/tests/test_poisson_loss.py
# coding: ascii
"""Test offline de ConstrainedPoissonLoss (rule 5). No necesita el modelo
completo -- construye tensores sinteticos que simulan la salida del
softplus (siempre >= 0) para verificar que la loss es finita en casos
limite (target=0, lambda cercano a 0) y que penaliza mas cuando la
prediccion esta mas lejos del target."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from train import ConstrainedPoissonLoss


def test_finite_when_target_is_zero():
    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    pred = torch.full((2, 19), 0.001)   # lambda chico pero > 0 (softplus nunca da 0 exacto)
    target = torch.zeros(2, 19)
    loss = criterion(pred, target)
    assert torch.isfinite(loss), loss
    print(f"[OK] loss finita con target=0: {loss.item():.4f}")


def test_penaliza_mas_lejos_del_target():
    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    target = torch.full((1, 19), 3.0)
    cerca = torch.full((1, 19), 3.0)
    lejos = torch.full((1, 19), 0.1)
    loss_cerca = criterion(cerca, target)
    loss_lejos = criterion(lejos, target)
    assert loss_cerca.item() < loss_lejos.item(), (loss_cerca.item(), loss_lejos.item())
    print(f"[OK] loss(cerca)={loss_cerca.item():.4f} < loss(lejos)={loss_lejos.item():.4f}")


def test_termino_de_suma_penaliza_desbalance_de_totales():
    criterion_con_suma = ConstrainedPoissonLoss(lambda_sum=0.5)
    criterion_sin_suma = ConstrainedPoissonLoss(lambda_sum=0.0)
    target = torch.zeros(1, 19); target[0, 0] = 5.0   # total real = 5
    pred = torch.zeros(1, 19); pred[0, 0] = 0.001; pred[0, 1] = 4.999  # mismo total (~5), clase distinta
    # ambas dan Poisson NLL similar por clase 0 (target=5 vs lambda=0.001, mal) pero
    # el termino de suma es chico (total predicho ~ total real). Solo lo comparamos
    # contra una prediccion con el mismo error por clase pero total MUY distinto.
    pred_mal_total = torch.zeros(1, 19); pred_mal_total[0, 0] = 0.001; pred_mal_total[0, 1] = 0.001
    loss_total_ok = criterion_con_suma(pred, target)
    loss_total_mal = criterion_con_suma(pred_mal_total, target)
    assert loss_total_mal.item() > loss_total_ok.item(), (loss_total_mal.item(), loss_total_ok.item())
    print(f"[OK] termino de suma penaliza el desbalance de totales: "
          f"ok={loss_total_ok.item():.4f} < mal={loss_total_mal.item():.4f}")


if __name__ == "__main__":
    test_finite_when_target_is_zero()
    test_penaliza_mas_lejos_del_target()
    test_termino_de_suma_penaliza_desbalance_de_totales()
    print("\n>>> TEST POISSON LOSS OK <<<")
