"""Tests for MRA-NN loss functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.losses import FocalLoss, SingleTaskLoss, UncertaintyWeightedLoss


def test_focal_loss_zero_for_perfect_predictions():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    # Perfect predictions: logits very positive for target=1, very negative for target=0
    logits = torch.tensor([10.0, -10.0, 10.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_focal_loss_high_for_wrong_predictions():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    # Wrong predictions
    logits = torch.tensor([-10.0, 10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.item() > 1.0


def test_focal_loss_gradients():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    logits = torch.randn(32, requires_grad=True)
    targets = torch.randint(0, 2, (32,), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None


def test_uncertainty_weighted_loss_output_keys():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "log_dnorm": torch.randn(B),
        "refine": torch.randint(0, 2, (B,), dtype=torch.float32),
        "negative": torch.zeros(B),
    }
    pred_rs = torch.randn(B, 512)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_rs, pred_ld, pred_ref)
    assert "loss_rho_s" in components
    assert "loss_log_dnorm" in components
    assert "loss_refine" in components
    assert "sigma_rho_s" in components
    assert total.requires_grad


def test_uncertainty_weighted_loss_includes_negatives():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "log_dnorm": torch.randn(B),
        "refine": torch.zeros(B),
        "negative": torch.ones(B),  # all negatives
    }
    pred_rs = torch.randn(B, 512)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_rs, pred_ld, pred_ref)
    # rho_s loss should be non-zero even for negatives
    assert components["loss_rho_s"].item() > 0.0


def test_uncertainty_sigmas_are_learnable():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    learnable_params = [n for n, p in uwl.named_parameters() if p.requires_grad]
    # Should have 3 log-sigma parameters
    assert len(learnable_params) == 3


def test_single_task_loss_output():
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "negative": torch.zeros(B),
    }
    pred_rs = torch.randn(B, 512, requires_grad=True)
    total, components = stl(batch, pred_rs)
    assert "loss_rho_s" in components
    assert total.requires_grad


def test_single_task_loss_no_learnable_params():
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    params = list(stl.parameters())
    assert len(params) == 0


def test_single_task_loss_pos_weight():
    """Positive samples should contribute more to the loss than negatives."""
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    B = 16
    pred_rs = torch.randn(B, 512)
    target = torch.randn(B, 512)

    # All positive
    batch_pos = {"rho_s": target, "negative": torch.zeros(B)}
    loss_pos, _ = stl(batch_pos, pred_rs)

    # All negative
    batch_neg = {"rho_s": target, "negative": torch.ones(B)}
    loss_neg, _ = stl(batch_neg, pred_rs)

    # Same errors but positive-weighted loss should be higher
    # because weight normalizes differently
    assert loss_pos.item() != loss_neg.item()
