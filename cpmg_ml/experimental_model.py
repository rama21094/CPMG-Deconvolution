from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExperimentalCPMGTransformer(nn.Module):
    """Masked variable-length Transformer for experimental CPMG de-J prediction."""

    def __init__(
        self,
        point_dim: int = 3,
        global_dim: int = 1,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
        )

    def forward(
        self,
        points: torch.Tensor,
        b0: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.point_encoder(points) + self.global_encoder(b0).unsqueeze(1)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        output = self.output_head(hidden)
        mean = output[..., 0]
        log_sigma = output[..., 1].clamp(min=-7.0, max=5.0)
        return mean, log_sigma


def masked_experimental_loss(
    mean: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    slope_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    variance = torch.exp(2.0 * log_sigma)
    nll = 0.5 * ((target - mean) ** 2 / variance + 2.0 * log_sigma)
    nll_loss = nll[mask].mean()

    pair_mask = mask[:, 1:] & mask[:, :-1]
    mean_slope = mean[:, 1:] - mean[:, :-1]
    target_slope = target[:, 1:] - target[:, :-1]
    if pair_mask.any():
        slope_loss = F.mse_loss(mean_slope[pair_mask], target_slope[pair_mask])
    else:
        slope_loss = mean.new_tensor(0.0)

    loss = nll_loss + slope_weight * slope_loss
    return loss, {"nll": float(nll_loss.detach()), "slope_loss": float(slope_loss.detach())}


def masked_experimental_metrics(
    mean_norm: torch.Tensor,
    log_sigma_norm: torch.Tensor,
    target_norm: torch.Tensor,
    measured_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    normalization: dict,
) -> dict[str, float]:
    r2_mean = float(normalization["r2_mean"])
    r2_std = float(normalization["r2_std"])
    pred = mean_norm * r2_std + r2_mean
    sigma = torch.exp(log_sigma_norm) * r2_std

    error = pred - target_raw
    identity_error = measured_raw - target_raw
    valid_error = error[mask]
    valid_identity_error = identity_error[mask]
    valid_sigma = sigma[mask]

    rmse = torch.sqrt(torch.mean(valid_error**2)).item()
    identity_rmse = torch.sqrt(torch.mean(valid_identity_error**2)).item()
    improvement = 0.0 if identity_rmse == 0 else 100.0 * (identity_rmse - rmse) / identity_rmse
    coverage_68 = torch.mean((torch.abs(valid_error) <= valid_sigma).float()).item()
    coverage_95 = torch.mean((torch.abs(valid_error) <= 1.96 * valid_sigma).float()).item()

    return {
        "rmse": rmse,
        "mae": torch.mean(torch.abs(valid_error)).item(),
        "max_abs": torch.max(torch.abs(valid_error)).item(),
        "identity_rmse": identity_rmse,
        "identity_improvement_pct": improvement,
        "mean_predicted_uncertainty": torch.mean(valid_sigma).item(),
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
    }
