from .config import TFTFixedConfig
from .data import build_processed_panel, ensure_processed_panel
from .dataset import TFTFixedDataModule, build_dataset, load_data, validate_data
from .loss import LossSpec, build_loss
from .model import build_tft_model, load_tft_model_from_checkpoint

__all__ = [
    "TFTFixedConfig",
    "TFTFixedDataModule",
    "LossSpec",
    "build_processed_panel",
    "build_dataset",
    "build_loss",
    "build_tft_model",
    "ensure_processed_panel",
    "fit_tft_fixed_model",
    "load_data",
    "load_tft_model_from_checkpoint",
    "validate_data",
]


def __getattr__(name: str):
    if name == "fit_tft_fixed_model":
        from .train import fit_tft_fixed_model

        return fit_tft_fixed_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
