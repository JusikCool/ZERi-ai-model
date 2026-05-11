import torch
import torch.nn as nn


class AdaptivePinballLoss(nn.Module):
    def __init__(
        self,
        quantiles: list[float] = None,
        vix_threshold: float = 0.0,
        vix_scale: float = 1.0,
        sigma_scale: float = 1.0,
        alpha_down: float = 1.0,
        beta_down: float = 1.0,
        alpha_up: float = 1.0,
        beta_up: float = 1.0,
        crossing_weight: float = 0.1,
    ):
        super().__init__()
        self.quantiles = quantiles or [0.05,0.10,0.15,0.20,0.25,0.30,
             0.35,0.40,0.45,0.50,0.55,
             0.60,0.65,0.70,0.75,0.80,
             0.85,0.90,0.95]
        self.vix_threshold = vix_threshold
        self.vix_scale = vix_scale
        self.sigma_scale = sigma_scale
        self.alpha_down = alpha_down
        self.beta_down = beta_down
        self.alpha_up = alpha_up
        self.beta_up = beta_up
        self.crossing_weight = crossing_weight

    def compute_lambda(
        self, vix: torch.Tensor, sigma: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        vix_excess = torch.clamp(vix - self.vix_threshold, min=0.0) / self.vix_scale
        sigma_term = torch.relu(sigma) / self.sigma_scale
        if quantile < 0.5:
            return 1.0 + self.alpha_down * vix_excess + self.beta_down * sigma_term
        elif quantile > 0.5:
            return 1.0 + self.alpha_up * vix_excess + self.beta_up * sigma_term
        else:
            return torch.ones_like(vix)

    def _pinball(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        error = y_true - y_pred
        return torch.where(
            error >= 0,
            quantile * error,
            (quantile - 1.0) * error,
        )

    def _crossing_penalty(self, y_pred: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=y_pred.device)
        for i in range(len(self.quantiles) - 1):
            penalty = penalty + torch.relu(y_pred[..., i] - y_pred[..., i + 1]).mean()
        return penalty

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        vix: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=y_pred.device)
        for i, q in enumerate(self.quantiles):
            lambda_t = self.compute_lambda(vix, sigma, q)
            pb = self._pinball(y_pred[..., i], y_true, q)
            total_loss = total_loss + (lambda_t * pb).mean()

        total_loss = total_loss / len(self.quantiles)
        total_loss = total_loss + self.crossing_weight * self._crossing_penalty(y_pred)
        return total_loss
