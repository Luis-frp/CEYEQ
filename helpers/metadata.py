import pandas as pd


def read_metadata_csv(csv_path, dtype=None, **kwargs):
    del kwargs
    if csv_path is None:
        raise ValueError("Informe o caminho do CSV.")
    return pd.read_csv(csv_path, dtype=dtype or {})
