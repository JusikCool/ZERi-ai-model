import torch
import torch.nn as nn


class AdaptivePinballLoss(nn.Module):
    """
    M4 모델용 확장 Adaptive Pinball Loss.

    기존 M3 의 scalar α, β 를 sector(volatility group) 별 lookup table 로 확장.

    [M3 호환 모드]
      alpha_down_by_group 같은 group dict 가 None 이면 → 모든 group 에 동일 scalar α 적용
      → 기존 M3 와 동일한 loss 값 (M4-GARCH only 케이스)

    [M4-Combined 모드]
      group dict 가 주어지면 → group 별 다른 α, β 적용
      → λ_t(s) = 1 + α_s · vix_excess + β_s · sigma_term

    forward 의 vol_group_idx 인자:
      None 이면 모든 sample 을 group 0 으로 가정 (M3 호환)
      tensor 면 sample 별 lookup
    """

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
        # ===== M4 추가 =====
        group_labels: list[str] = None,
        alpha_down_by_group: dict = None,
        beta_down_by_group: dict = None,
        alpha_up_by_group: dict = None,
        beta_up_by_group: dict = None,
    ):
        super().__init__()
        self.quantiles = quantiles or [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95,
        ]
        self.vix_threshold = vix_threshold
        self.vix_scale = vix_scale
        self.sigma_scale = sigma_scale

        # Scalar fallback (M3 호환)
        self.alpha_down = alpha_down
        self.beta_down = beta_down
        self.alpha_up = alpha_up
        self.beta_up = beta_up
        self.crossing_weight = crossing_weight

        # ===== M4: group-level coefficient tables =====
        self.group_labels = group_labels or ["low_vol", "mid_vol", "high_vol"]
        n_groups = len(self.group_labels)

        # 각 group 별 table 을 buffer 로 등록 (학습 안 됨, gpu 자동 이동)
        self.register_buffer(
            "alpha_down_table",
            self._build_table(alpha_down_by_group, alpha_down, n_groups),
        )
        self.register_buffer(
            "beta_down_table",
            self._build_table(beta_down_by_group, beta_down, n_groups),
        )
        self.register_buffer(
            "alpha_up_table",
            self._build_table(alpha_up_by_group, alpha_up, n_groups),
        )
        self.register_buffer(
            "beta_up_table",
            self._build_table(beta_up_by_group, beta_up, n_groups),
        )

    def _build_table(self, group_dict, scalar_default, n_groups):
        """dict 가 None 이면 모든 group 에 scalar_default. 아니면 dict lookup."""
        if group_dict is None:
            return torch.full((n_groups,), float(scalar_default), dtype=torch.float32)
        # dict keys 가 self.group_labels 와 일치해야 함
        values = []
        for label in self.group_labels:
            if label not in group_dict:
                raise KeyError(
                    f"group_labels 와 dict key 불일치: '{label}' 없음. "
                    f"dict keys = {list(group_dict.keys())}, "
                    f"group_labels = {self.group_labels}"
                )
            values.append(float(group_dict[label]))
        return torch.tensor(values, dtype=torch.float32)

    def compute_lambda(
        self,
        vix: torch.Tensor,
        sigma: torch.Tensor,
        quantile: float,
        vol_group_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        λ_t (batch, time) 계산.

        Args:
            vix: (batch, time)
            sigma: (batch, time)
            quantile: 분위수 q (scalar)
            vol_group_idx: (batch,) long tensor, 각 sample 의 group index (0/1/2)

        Returns:
            (batch, time) tensor of λ_t
        """
        vix_excess = torch.clamp(vix - self.vix_threshold, min=0.0) / self.vix_scale
        sigma_term = torch.relu(sigma) / self.sigma_scale

        if quantile < 0.5:
            alpha_s = self.alpha_down_table[vol_group_idx].unsqueeze(-1)  # (batch, 1)
            beta_s = self.beta_down_table[vol_group_idx].unsqueeze(-1)
            return 1.0 + alpha_s * vix_excess + beta_s * sigma_term
        elif quantile > 0.5:
            alpha_s = self.alpha_up_table[vol_group_idx].unsqueeze(-1)
            beta_s = self.beta_up_table[vol_group_idx].unsqueeze(-1)
            return 1.0 + alpha_s * vix_excess + beta_s * sigma_term
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
        vol_group_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (batch, time, n_quantiles)
            y_true: (batch, time)
            vix:    (batch, time)
            sigma:  (batch, time)
            vol_group_idx: (batch,) long tensor. None 이면 모든 sample 을 group 0 으로.
                           M3 호환 (M4-GARCH only) 시 None 또는 전달 안 함.

        Returns:
            scalar loss
        """
        if vol_group_idx is None:
            # M3 호환 / M4-GARCH only: 모든 sample 을 group 0 으로
            vol_group_idx = torch.zeros(
                y_pred.shape[0], dtype=torch.long, device=y_pred.device
            )

        total_loss = torch.tensor(0.0, device=y_pred.device)
        for i, q in enumerate(self.quantiles):
            lambda_t = self.compute_lambda(vix, sigma, q, vol_group_idx)
            pb = self._pinball(y_pred[..., i], y_true, q)
            total_loss = total_loss + (lambda_t * pb).mean()

        total_loss = total_loss / len(self.quantiles)
        total_loss = total_loss + self.crossing_weight * self._crossing_penalty(y_pred)
        return total_loss
