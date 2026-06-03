import os

from helpers.dataset import StomachDataset
from helpers.metadata import read_metadata_csv

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold


DEFAULT_TRAIN_CSV = os.path.join("data", "new_train_eyeq_v2.csv")
DEFAULT_TEST_CSV = os.path.join("data", "teste_processado.csv")
DEFAULT_DATA_ROOT = os.path.join("data", "images")
DEFAULT_IMAGE_COLUMN = "image"
DEFAULT_LABEL_COLUMN = "quality"


def _resolve_transform(transform, default_image_size=(224, 224)):
    if transform is not None:
        return transform
    return transforms.Compose([
        transforms.Resize(default_image_size),
        transforms.ToTensor(),
    ])


def tranform_train():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
    ])


def _resolve_column(df, requested_column, default_column, column_description):
    if requested_column in df.columns:
        return requested_column
    if default_column in df.columns:
        return default_column
    raise ValueError(
        f"O CSV precisa conter a coluna '{requested_column}'"
        f" ou '{default_column}' para {column_description}."
    )


def _read_split_source(csv_path, image_column, label_column):
    if csv_path is None:
        raise ValueError("Informe o caminho do CSV.")
    df = read_metadata_csv(csv_path, dtype={image_column: str})
    image_column = _resolve_column(df, image_column, DEFAULT_IMAGE_COLUMN, "imagem")
    label_column = _resolve_column(df, label_column, DEFAULT_LABEL_COLUMN, "rotulo")
    return df, image_column, label_column


def validate_no_data_leakage(fold_idx, train_indices, val_indices):
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise ValueError(
            f"Vazamento de dados detectado no fold {fold_idx}. "
            f"Amostras repetidas entre treino e validacao: {sorted(overlap)[:10]}"
        )


def validate_non_empty_splits(fold_idx, train_dataset, val_dataset, test_dataset):
    split_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    empty_splits = [split_name for split_name, split_size in split_sizes.items() if split_size == 0]
    if empty_splits:
        raise ValueError(f"Split vazio no fold {fold_idx}: {split_sizes}.")


def division_of_groups(
    csv_metadata=DEFAULT_TRAIN_CSV,
    num_folds=5,
    random_state=42,
    filepath_column=DEFAULT_IMAGE_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
    split_csv_path=None,
    class_name_column=None,
):
    del split_csv_path, class_name_column

    df, _, label_column = _read_split_source(csv_metadata, filepath_column, label_column)
    labels = df[label_column].astype(int).to_numpy()
    row_indices = df.index.to_numpy()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(row_indices, labels)):
        train_rows = row_indices[train_indices].tolist()
        val_rows = row_indices[val_indices].tolist()
        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_rows,
                "val_indices": val_rows,
            }
        )

    return folds


def get_fold_datasets(
    fold_idx,
    transform_train=None,
    transform_eval=None,
    csv_metadata=DEFAULT_TRAIN_CSV,
    test_csv=DEFAULT_TEST_CSV,
    data_root=DEFAULT_DATA_ROOT,
    split_csv_path=None,
    num_folds=5,
    random_state=42,
    filepath_column=DEFAULT_IMAGE_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
    class_name_column=None,
):
    del split_csv_path, class_name_column
    test_csv = test_csv or DEFAULT_TEST_CSV

    train_folds = division_of_groups(
        csv_metadata=csv_metadata,
        num_folds=num_folds,
        random_state=random_state,
        filepath_column=filepath_column,
        label_column=label_column,
    )

    if fold_idx < 0 or fold_idx >= len(train_folds):
        raise ValueError(f"fold_idx deve estar entre 0 e {len(train_folds) - 1}.")

    fold_data = train_folds[fold_idx]
    validate_no_data_leakage(fold_idx, fold_data["train_indices"], fold_data["val_indices"])
    transform_train = _resolve_transform(transform_train)
    transform_eval = _resolve_transform(transform_eval)

    dataset_kwargs = {
        "root": data_root,
        "filepath_column": filepath_column,
        "label_column": label_column,
    }

    train_dataset = StomachDataset(
        csv_path=csv_metadata,
        transform=transform_train,
        allowed_indices=fold_data["train_indices"],
        **dataset_kwargs,
    )

    val_dataset = StomachDataset(
        csv_path=csv_metadata,
        transform=transform_eval,
        allowed_indices=fold_data["val_indices"],
        **dataset_kwargs,
    )

    test_dataset = StomachDataset(
        csv_path=test_csv,
        transform=transform_eval,
        **dataset_kwargs,
    )

    validate_non_empty_splits(fold_idx, train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset


def get_fold_dataloaders(
    fold_idx,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    persistent_workers=True,
    prefetch_factor=2,
    transform_train=tranform_train(),
    transform_eval=None,
    csv_metadata=DEFAULT_TRAIN_CSV,
    test_csv=DEFAULT_TEST_CSV,
    data_root=DEFAULT_DATA_ROOT,
    split_csv_path=None,
    num_folds=5,
    random_state=42,
    filepath_column=DEFAULT_IMAGE_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
    class_name_column=None,
):
    train_dataset, val_dataset, test_dataset = get_fold_datasets(
        fold_idx=fold_idx,
        transform_train=transform_train,
        transform_eval=transform_eval,
        csv_metadata=csv_metadata,
        test_csv=test_csv,
        data_root=data_root,
        split_csv_path=split_csv_path,
        num_folds=num_folds,
        random_state=random_state,
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
