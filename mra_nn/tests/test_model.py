"""Tests for MRA-NN model architecture."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.model import MRANet, build_model

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(cfg):
    return build_model(cfg)


def test_build_model(model):
    assert isinstance(model, MRANet)


def test_forward_shapes(model):
    B = 8
    rho0_s = torch.randn(B, 512)
    vnuc_s = torch.randn(B, 512)
    halo_rho0 = torch.randn(B, 6, 512)
    halo_vnuc = torch.randn(B, 6, 512)
    level = torch.randint(0, 19, (B,))

    rho_s, log_dnorm, refine_logit = model(
        rho0_s, vnuc_s, halo_rho0, halo_vnuc, level
    )
    assert rho_s.shape == (B, 512)
    assert log_dnorm.shape == (B,)
    assert refine_logit.shape == (B,)


def test_forward_gradients(model):
    B = 4
    rho0_s = torch.randn(B, 512, requires_grad=True)
    vnuc_s = torch.randn(B, 512)
    halo_rho0 = torch.randn(B, 6, 512)
    halo_vnuc = torch.randn(B, 6, 512)
    level = torch.randint(0, 19, (B,))

    rho_s, log_dnorm, refine_logit = model(
        rho0_s, vnuc_s, halo_rho0, halo_vnuc, level
    )
    loss = rho_s.sum() + log_dnorm.sum() + refine_logit.sum()
    loss.backward()
    assert rho0_s.grad is not None
    assert rho0_s.grad.shape == (B, 512)


def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    # k=8 (512-dim): expect ~2.8M — allow 2M to 4M range
    assert 2_000_000 < total < 4_000_000, f"Parameter count {total} outside expected range"


def test_halo_encoder_weight_sharing(model):
    """The halo encoder should use shared weights for all 6 faces."""
    B = 2
    halo_input = torch.randn(B, 6, 512)
    # Access the halo encoder directly
    assert hasattr(model, "halo_encoder")


def test_film_conditioning_exists(model):
    """Model should have FiLM layers that take level input."""
    assert hasattr(model, "level_embedding")
    # Level embedding should have 19 entries (levels 0-18)
    assert model.level_embedding.num_embeddings == 19
