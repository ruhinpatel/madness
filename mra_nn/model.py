"""MRANet — FiLM-conditioned MLP with factored halo encoder.

Architecture (from design spec, k=8):
  - Halo encoder: shared MLP across 6 face-adjacent neighbors (1032 -> 256 -> 128)
  - Center features: rho0_s(512) + vnuc_s(512) = 1024
  - Trunk: 3 FiLM-conditioned layers (1792 -> 1024 -> 512 -> 256)
  - FiLM: level embedding (32-dim) -> per-layer gamma/beta
  - Heads: rho_s (256->512), log_dnorm (256->1), refine (256->1)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HaloEncoder(nn.Module):
    """Shared encoder for face-adjacent neighbor boxes.

    Processes each of 6 neighbors identically through a shared MLP,
    with a learnable face embedding to distinguish +x/-x/+y/-y/+z/-z.
    """

    def __init__(
        self,
        k_cubed: int = 512,
        face_embed_dim: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 128,
        n_faces: int = 6,
    ) -> None:
        super().__init__()
        self.n_faces = n_faces
        self.face_embedding = nn.Embedding(n_faces, face_embed_dim)
        # Input: rho0_halo(k^3) + vnuc_halo(k^3) + face_embed
        input_dim = 2 * k_cubed + face_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, halo_rho0: torch.Tensor, halo_vnuc: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        halo_rho0 : Tensor [B, 6, k^3]
        halo_vnuc : Tensor [B, 6, k^3]

        Returns
        -------
        Tensor [B, 6 * out_dim]
            Concatenated halo embeddings.
        """
        B = halo_rho0.shape[0]
        # Face indices [0..5], expanded to [B, 6]
        face_ids = torch.arange(self.n_faces, device=halo_rho0.device)
        face_ids = face_ids.unsqueeze(0).expand(B, -1)  # [B, 6]
        face_emb = self.face_embedding(face_ids)  # [B, 6, face_embed_dim]

        # Concatenate per-face: [B, 6, 2*k^3 + face_embed_dim]
        x = torch.cat([halo_rho0, halo_vnuc, face_emb], dim=-1)

        # Reshape to process all faces at once through shared MLP
        B, F, D = x.shape
        x = x.reshape(B * F, D)  # [B*6, input_dim]
        x = self.mlp(x)  # [B*6, out_dim]
        x = x.reshape(B, F, -1)  # [B, 6, out_dim]

        # Concatenate face outputs
        return x.reshape(B, -1)  # [B, 6*out_dim]


class FiLMLayer(nn.Module):
    """Linear -> BatchNorm -> FiLM(gamma, beta) -> ReLU -> Dropout.

    FiLM conditioning: output = gamma * BatchNorm(Wx + b) + beta
    where gamma and beta are produced from the level embedding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        level_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        # Project level embedding to (gamma, beta) pair
        self.film_proj = nn.Linear(level_embed_dim, 2 * out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, level_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_dim]
        level_emb : Tensor [B, level_embed_dim]

        Returns
        -------
        Tensor [B, out_dim]
        """
        h = self.linear(x)
        h = self.bn(h)
        # FiLM modulation
        film = self.film_proj(level_emb)  # [B, 2*out_dim]
        gamma, beta = film.chunk(2, dim=-1)  # each [B, out_dim]
        h = gamma * h + beta
        return self.dropout(self.relu(h))


class MRANet(nn.Module):
    """FiLM-conditioned MLP with factored halo encoder for MRA-NN.

    Three output heads:
      - rho_s: density coefficients [B, k^3]
      - log_dnorm: wavelet norm [B]
      - refine: refinement logit [B] (apply sigmoid externally for probabilities)
    """

    def __init__(
        self,
        k_cubed: int = 512,
        n_faces: int = 6,
        n_levels: int = 19,
        level_embed_dim: int = 32,
        face_embed_dim: int = 8,
        halo_encoder_hidden: int = 256,
        halo_encoder_out: int = 128,
        trunk_dims: tuple = (1024, 512, 256),
        dropout: float = 0.1,
        single_task: bool = False,
        use_parent_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_parent_features = use_parent_features

        # --- Halo encoder (shared across 6 faces) ---
        self.halo_encoder = HaloEncoder(
            k_cubed=k_cubed,
            face_embed_dim=face_embed_dim,
            hidden_dim=halo_encoder_hidden,
            out_dim=halo_encoder_out,
            n_faces=n_faces,
        )

        # --- Level embedding (for FiLM conditioning) ---
        self.level_embedding = nn.Embedding(n_levels, level_embed_dim)

        # --- Trunk MLP with FiLM conditioning ---
        # Input: halo_encoder output (6*128=768) + center features (2*512=1024)
        #        + optional parent features (2*512=1024)
        center_dim = 2 * k_cubed  # rho0_s + vnuc_s
        parent_dim = 2 * k_cubed if use_parent_features else 0
        halo_out_dim = n_faces * halo_encoder_out
        trunk_input_dim = halo_out_dim + center_dim + parent_dim

        trunk_layers = []
        in_dim = trunk_input_dim
        for out_dim in trunk_dims:
            trunk_layers.append(
                FiLMLayer(in_dim, out_dim, level_embed_dim, dropout)
            )
            in_dim = out_dim
        self.trunk = nn.ModuleList(trunk_layers)

        # --- Output heads ---
        final_dim = trunk_dims[-1]  # 256
        self.single_task = single_task
        self.head_rho_s = nn.Linear(final_dim, k_cubed)
        if not single_task:
            self.head_log_dnorm = nn.Linear(final_dim, 1)
            nn.init.constant_(self.head_log_dnorm.bias, -27.5)
            self.head_refine = nn.Linear(final_dim, 1)

    def forward(
        self,
        rho0_s: torch.Tensor,
        vnuc_s: torch.Tensor,
        halo_rho0: torch.Tensor,
        halo_vnuc: torch.Tensor,
        level: torch.Tensor,
        parent_rho0_s: torch.Tensor | None = None,
        parent_vnuc_s: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rho0_s         : [B, k^3]
        vnuc_s         : [B, k^3]
        halo_rho0      : [B, 6, k^3]
        halo_vnuc      : [B, 6, k^3]
        level          : [B] (long)
        parent_rho0_s  : [B, k^3] or None (required if use_parent_features=True)
        parent_vnuc_s  : [B, k^3] or None (required if use_parent_features=True)

        Returns
        -------
        rho_s        : [B, k^3]
        log_dnorm    : [B]
        refine_logit : [B]
        """
        # Halo encoding
        halo_emb = self.halo_encoder(halo_rho0, halo_vnuc)  # [B, 768]

        # Center features
        center = torch.cat([rho0_s, vnuc_s], dim=-1)  # [B, 1024]

        # Level embedding (for FiLM, not concatenated)
        level_emb = self.level_embedding(level)  # [B, 32]

        # Trunk input
        if self.use_parent_features:
            parent = torch.cat([parent_rho0_s, parent_vnuc_s], dim=-1)  # [B, 1024]
            x = torch.cat([halo_emb, center, parent], dim=-1)  # [B, 2816]
        else:
            x = torch.cat([halo_emb, center], dim=-1)  # [B, 1792]

        # FiLM-conditioned trunk
        for film_layer in self.trunk:
            x = film_layer(x, level_emb)

        # Output heads — rho_s uses residual from rho0_s so the model only
        # needs to learn the small correction (rho_s - rho0_s).
        rho_s = self.head_rho_s(x) + rho0_s  # [B, 512]
        if self.single_task:
            return rho_s, None, None
        log_dnorm = self.head_log_dnorm(x).squeeze(-1)  # [B]
        refine_logit = self.head_refine(x).squeeze(-1)  # [B]

        return rho_s, log_dnorm, refine_logit


def build_model(cfg: dict) -> MRANet:
    """Construct MRANet from config dict."""
    m = cfg["model"]
    return MRANet(
        k_cubed=m["k_cubed"],
        n_faces=m["n_faces"],
        n_levels=m["n_levels"],
        level_embed_dim=m["level_embed_dim"],
        face_embed_dim=m["face_embed_dim"],
        halo_encoder_hidden=m["halo_encoder_hidden"],
        halo_encoder_out=m["halo_encoder_out"],
        trunk_dims=tuple(m["trunk_dims"]),
        dropout=m["dropout"],
        single_task=m.get("single_task", False),
        use_parent_features=m.get("use_parent_features", False),
    )
