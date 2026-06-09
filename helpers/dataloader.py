import os

from helpers.dataset import EyeQDataset, extract_patient_id
from helpers.metadata import read_metadata_csv

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold


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


# def tranform_train():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(180),
#         transforms.ToTensor(),
#     ])


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


def validate_no_data_leakage(fold_idx, train_indices, val_indices, train_groups=None, val_groups=None):
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise ValueError(
            f"Vazamento de dados detectado no fold {fold_idx}. "
            f"Amostras repetidas entre treino e validacao: {sorted(overlap)[:10]}"
        )

    if train_groups is not None and val_groups is not None:
        train_groups = set(train_groups)
        val_groups = set(val_groups)
        group_overlap = train_groups & val_groups
        if group_overlap:
            raise ValueError(
                f"Vazamento de pacientes detectado no fold {fold_idx}. "
                f"Pacientes repetidos entre treino e validacao: {sorted(group_overlap)[:10]}"
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

    df, filepath_column, label_column = _read_split_source(csv_metadata, filepath_column, label_column)
    labels = df[label_column].astype(int).to_numpy()
    groups = df[filepath_column].map(extract_patient_id).to_numpy()
    row_indices = df.index.to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(df.index, labels, groups)):
        train_rows = row_indices[train_indices].tolist()
        val_rows = row_indices[val_indices].tolist()
        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_rows,
                "val_indices": val_rows,
                "train_groups": sorted(set(groups[train_indices].tolist())),
                "val_groups": sorted(set(groups[val_indices].tolist())),
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
    test_data_root=None,
    split_csv_path=None,
    num_folds=5,
    random_state=42,
    filepath_column=DEFAULT_IMAGE_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
    class_name_column=None,
):
    del split_csv_path, class_name_column
    test_csv = test_csv or DEFAULT_TEST_CSV
    test_data_root = test_data_root or data_root

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
    validate_no_data_leakage(
        fold_idx,
        fold_data["train_indices"],
        fold_data["val_indices"],
        train_groups=fold_data.get("train_groups"),
        val_groups=fold_data.get("val_groups"),
    )
    transform_train = _resolve_transform(transform_train)
    transform_eval = _resolve_transform(transform_eval)

    dataset_kwargs = {
        "root": data_root,
        "filepath_column": filepath_column,
        "label_column": label_column,
    }

    train_dataset = EyeQDataset(
        root=data_root,
        transform=transform_train,
        csv_path=csv_metadata,
        allowed_indices=fold_data["train_indices"],
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )
    val_dataset = EyeQDataset(
        root=data_root,
        transform=transform_eval,
        csv_path=csv_metadata,
        allowed_indices=fold_data["val_indices"],
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )

    test_dataset = EyeQDataset(
        root=test_data_root,
        csv_path=test_csv,
        transform=transform_eval,
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )

    validate_non_empty_splits(fold_idx, train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset


def get_fold_dataloaders(
    fold_idx,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    transform_train=None,
    transform_eval=None,
    csv_metadata=DEFAULT_TRAIN_CSV,
    test_csv=DEFAULT_TEST_CSV,
    data_root=DEFAULT_DATA_ROOT,
    test_data_root=None,
    num_folds=5,
    random_state=42,
    filepath_column=DEFAULT_IMAGE_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
    class_name_column=None,
    split_csv_path=None,
    collate_fn=None,
):
    transform_train = _resolve_transform(transform_train)
    transform_eval = _resolve_transform(transform_eval)

    test_data_root = test_data_root or data_root

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
    validate_no_data_leakage(
        fold_idx,
        fold_data["train_indices"],
        fold_data["val_indices"],
        train_groups=fold_data.get("train_groups"),
        val_groups=fold_data.get("val_groups"),
    )

    dataset_kwargs = {
        "root": data_root,
        "filepath_column": filepath_column,
        "label_column": label_column,
    }

    train_dataset = EyeQDataset(
        root=data_root,
        transform=transform_train,
        csv_path=csv_metadata,
        allowed_indices=fold_data["train_indices"],
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )
    val_dataset = EyeQDataset(
        root=data_root,
        transform=transform_eval,
        csv_path=csv_metadata,
        allowed_indices=fold_data["val_indices"],
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )

    test_dataset = EyeQDataset(
        root=test_data_root,
        csv_path=test_csv,
        transform=transform_eval,
        filepath_column=filepath_column,
        label_column=label_column,
        class_name_column=class_name_column,
    )

    validate_non_empty_splits(fold_idx, train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
