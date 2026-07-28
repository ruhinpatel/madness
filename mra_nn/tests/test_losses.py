"""Tests for MRA-NN loss functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.losses import FocalLoss, UncertaintyWeightedLoss


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
        "delta_rho": torch.randn(B, 216),
        "log_dnorm": torch.randn(B),
        "refine": torch.randint(0, 2, (B,), dtype=torch.float32),
        "negative": torch.zeros(B),
    }
    pred_dr = torch.randn(B, 216)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_dr, pred_ld, pred_ref)
    assert "loss_delta_rho" in components
    assert "loss_log_dnorm" in components
    assert "loss_refine" in components
    assert "sigma_delta_rho" in components
    assert total.requires_grad


def test_uncertainty_weighted_loss_masks_negatives():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    # All negatives — delta_rho loss should be zero
    batch = {
        "delta_rho": torch.randn(B, 216),
        "log_dnorm": torch.randn(B),
        "refine": torch.zeros(B),
        "negative": torch.ones(B),  # all negatives
    }
    pred_dr = torch.randn(B, 216)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_dr, pred_ld, pred_ref)
    assert components["loss_delta_rho"].item() == 0.0


def test_uncertainty_sigmas_are_learnable():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    learnable_params = [n for n, p in uwl.named_parameters() if p.requires_grad]
    # Should have 3 log-sigma parameters
    assert len(learnable_params) == 3
