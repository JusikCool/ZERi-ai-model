"""
M4 모델 모듈
============

본 모델은 두 선행 연구를 본 연구의 구조에 흡수한 확장 버전:

- Frank (2023) 의 sectoral approach → Adaptive Pinball Loss 의 group 별 가중치 차등
- Buczyński & Chlebus (2024) GARCHNet 의 hybrid σ_t → 입력 feature + loss σ 양쪽 활용

두 가지 실험 모드:

- M4-GARCH only: σ 자리 GARCH_Variance + scalar α/β
- M4-Combined:   σ 자리 GARCH_Variance + group 별 차등 α/β

학습 ablation 으로 두 모드의 효과를 분리 측정.
"""

from .model import M4FullModel
from .loss import AdaptivePinballLoss

__all__ = ["M4FullModel", "AdaptivePinballLoss"]
