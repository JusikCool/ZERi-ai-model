try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover - fallback for older environments
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

__all__ = [
    "pl",
    "EarlyStopping",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "CSVLogger",
]
