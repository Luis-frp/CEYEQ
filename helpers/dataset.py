import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from helpers.metadata import read_metadata_csv


def extract_patient_id(image_path):
    filename = os.path.basename(str(image_path).replace("\\", "/"))
    stem, _ = os.path.splitext(filename)
    return stem.split("_", 1)[0]


class EyeQDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        csv_path=None,
        allowed_indices=None,
        filepath_column="image",
        label_column="quality",
        class_name_column=None,
    ):
        del class_name_column

        self.root = root
        self.transform = transform
        self.filepath_column = filepath_column
        self.label_column = label_column

        df = read_metadata_csv(csv_path, dtype={filepath_column: str})
        filepath_column = self._resolve_column(df, filepath_column, "image", "imagem")
        label_column = self._resolve_column(df, label_column, "quality", "rotulo")
        self.filepath_column = filepath_column
        self.label_column = label_column

        allowed_indices = set(int(idx) for idx in allowed_indices) if allowed_indices is not None else None
        self.samples = []
        self.metadata = []

        for row_index, row in df.iterrows():
            if allowed_indices is not None and int(row_index) not in allowed_indices:
                continue

            rel_path = str(row[filepath_column]).replace("\\", "/").strip()
            patient_id = extract_patient_id(rel_path)
            label = int(row[label_column])
            img_path = self._resolve_image_path(rel_path)

            self.samples.append((img_path, label))
            self.metadata.append(
                {
                    "row_index": int(row_index),
                    "patient_id": patient_id,
                    "image": rel_path,
                    "quality": label,
                }
            )

    @staticmethod
    def _resolve_column(df, requested_column, default_column, column_description):
        if requested_column in df.columns:
            return requested_column
        if default_column in df.columns:
            return default_column
        raise ValueError(
            f"O CSV precisa conter a coluna '{requested_column}'"
            f" ou '{default_column}' para {column_description}."
        )

    def _resolve_image_path(self, rel_path):
        direct_path = os.path.join(self.root, rel_path)
        if os.path.exists(direct_path):
            return direct_path

        base_path, _ = os.path.splitext(rel_path)
        for ext in [".jpeg", ".jpg", ".png"]:
            alt_path = os.path.join(self.root, base_path + ext)
            if os.path.exists(alt_path):
                return alt_path

        return direct_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError:
            print(f"Aviso: Arquivo de imagem não encontrado, pulando: {path}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
