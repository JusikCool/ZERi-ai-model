from model.sector_models_m4.dataset import (
    build_sector_crisis_dataloaders,
    build_sector_dataloaders,
    build_sector_dataset,
    load_sector_data,
)

SECTOR_ID = "energy"


def load():
    return load_sector_data(SECTOR_ID)


def build():
    return build_sector_dataset(SECTOR_ID)


def dataloaders(train_ds, val_ds, batch_size: int = 64):
    return build_sector_dataloaders(train_ds, val_ds, batch_size=batch_size)


def crisis_dataloaders(train_ds, val_ds, df, encoder_length: int, batch_size: int = 64):
    return build_sector_crisis_dataloaders(
        train_ds, val_ds, df,
        encoder_length=encoder_length,
        batch_size=batch_size,
    )
