"""Loss functions for MRA-NN multi-task training.

- FocalLoss: focal binary cross-entropy for the refine head (Lin et al. 2017)
- UncertaintyWeightedLoss: Kendall et al. 2018 uncertainty-weighted multi-task loss
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017).

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) for target=1, else 1 - sigmoid(logit).

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 recovers standard BCE.
    alpha : float
        Weight for the positive class (target=1).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [N] raw logits (before sigmoid)
        targets : [N] float, 0.0 or 1.0

        Returns
        -------
        Scalar mean focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # equals sigmoid(logit) when target=1, etc.
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        return (alpha_weight * focal_weight * bce).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al. 2018).

    L = (1/2*sigma1^2)*L_rs + (1/2*sigma2^2)*L_ld + (1/sigma3^2)*L_ref
        + log(sigma1*sigma2*sigma3)

    Three learnable log-sigma parameters auto-balance the task weights.
    """

    def __init__(self, focal_gamma: float = 2.0, focal_alpha: float = 0.75) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        # Learnable log-variance parameters, initialized to 0 (sigma=1)
        self.log_sigma_rs = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ld = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ref = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        pred_rho_s: torch.Tensor,
        pred_log_dnorm: torch.Tensor,
        pred_refine_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch : dict with keys rho_s, log_dnorm, refine, negative
        pred_rho_s        : [B, 512]
        pred_log_dnorm    : [B]
        pred_refine_logit : [B]

        Returns
        -------
        total_loss : scalar
        components : dict of individual losses and sigma values (detached)
        """
        # --- rho_s MSE (all samples) ---
        loss_rs = F.mse_loss(pred_rho_s, batch["rho_s"])

        # --- Log-dnorm MSE (all samples) ---
        loss_ld = F.mse_loss(pred_log_dnorm, batch["log_dnorm"])

        # --- Refine focal loss (all samples) ---
        loss_ref = self.focal_loss(pred_refine_logit, batch["refine"])

        # --- Uncertainty weighting ---
        sigma_rs = torch.exp(self.log_sigma_rs)
        sigma_ld = torch.exp(self.log_sigma_ld)
        sigma_ref = torch.exp(self.log_sigma_ref)

        total = (
            0.5 / sigma_rs**2 * loss_rs
            + 0.5 / sigma_ld**2 * loss_ld
            + 1.0 / sigma_ref**2 * loss_ref
            + self.log_sigma_rs + self.log_sigma_ld + self.log_sigma_ref
        )

        components = {
            "loss_rho_s": loss_rs.detach(),
            "loss_log_dnorm": loss_ld.detach(),
            "loss_refine": loss_ref.detach(),
            "sigma_rho_s": sigma_rs.detach(),
            "sigma_log_dnorm": sigma_ld.detach(),
            "sigma_refine": sigma_ref.detach(),
        }
        return total, components
