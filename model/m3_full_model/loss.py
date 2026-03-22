import torch
import torch.nn as nn


class AdaptivePinballLoss(nn.Module):
    def __init__(
        self,
        quantiles: list[float] = None,
        vix_threshold: float = 25.0,
        vix_scale: float = 25.0,
        sigma_scale: float = 0.02,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.vix_threshold = vix_threshold
        self.vix_scale = vix_scale
        self.sigma_scale = sigma_scale
        self.alpha = alpha
        self.beta = beta

    def compute_lambda(
        self, vix: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        vix_excess = torch.clamp(vix - self.vix_threshold, min=0.0) / self.vix_scale
        sigma_term = sigma / self.sigma_scale
        return 1.0 + self.alpha * vix_excess + self.beta * sigma_term

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

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        vix: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        lambda_t = self.compute_lambda(vix, sigma)

        total_loss = torch.tensor(0.0, device=y_pred.device)
        for i, q in enumerate(self.quantiles):
            pb = self._pinball(y_pred[..., i], y_true, q)
            total_loss = total_loss + (lambda_t * pb).mean()

        return total_loss / len(self.quantiles)
