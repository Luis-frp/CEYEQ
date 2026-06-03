import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.dataloader import get_fold_datasets
from models.generic_model import get_model
from utils.config import (
    _apply_environment,
    _apply_local_data_cache,
    _expand_config_values,
    _infer_dataset_classes,
    build_transforms,
)

import yaml


def _load_runtime_config(config_path=None):
    config_path = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file) or {}

    _apply_environment(config_dict)
    config_dict = _expand_config_values(config_dict)

    for path_key in [
        "data_dir",
        "metadata_csv",
        "test_csv",
        "split_csv_path",
        "results_dir",
        "viz_dir",
        "local_cache_dir",
    ]:
        if config_dict.get(path_key) and not os.path.isabs(config_dict[path_key]):
            config_dict[path_key] = os.path.join(PROJECT_ROOT, config_dict[path_key])

    config_dict.setdefault("image_size", [224, 224])
    config_dict.setdefault("num_folds", 5)
    config_dict.setdefault("split_random_state", 42)
    config_dict.setdefault("num_classes", 3)
    config_dict.setdefault("batch_size", 32)
    config_dict.setdefault("num_workers", 0)
    config_dict.setdefault("pin_memory", True)
    config_dict.setdefault("persistent_workers", True)
    config_dict.setdefault("prefetch_factor", 2)
    config_dict.setdefault("data_dir", os.path.join(PROJECT_ROOT, "data", "images"))
    config_dict.setdefault("metadata_csv", os.path.join(PROJECT_ROOT, "data", "new_train_eyeq_v2.csv"))
    config_dict.setdefault("test_csv", os.path.join(PROJECT_ROOT, "data", "teste_processado.csv"))
    config_dict.setdefault("split_csv_path", None)
    config_dict.setdefault("viz_dir", os.path.join(PROJECT_ROOT, "vizualizacoes"))
    config_dict.setdefault("results_dir", os.path.join(PROJECT_ROOT, "resultados"))
    config_dict.setdefault("filepath_column", "image")
    config_dict.setdefault("label_column", "quality")
    config_dict.setdefault("class_name_column", None)
    config_dict.setdefault("local_cache_dir", None)

    _apply_local_data_cache(config_dict)
    _infer_dataset_classes(config_dict)
    return config_dict


def _safe_model_tag(model_name, version):
    model_tag = model_name if version is None else f"{model_name}.{version}"
    return model_tag, model_tag.replace("/", "_")


def _resolve_weight_path(weights_dir, weight_file, model_name, version, fold_idx):
    if weight_file:
        weight_path = weight_file
        if not os.path.isabs(weight_path):
            weight_path = os.path.join(PROJECT_ROOT, weight_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Peso nao encontrado: {weight_path}")
        return weight_path

    if not weights_dir:
        raise ValueError("Informe --weights-dir ou --weight-file.")

    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(PROJECT_ROOT, weights_dir)
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Pasta de pesos nao encontrada: {weights_dir}")

    _, safe_model_tag = _safe_model_tag(model_name, version)
    expected_name = f"{safe_model_tag}_fold_{fold_idx}.pth"
    expected_path = os.path.join(weights_dir, expected_name)
    if os.path.exists(expected_path):
        return expected_path

    weight_files = [
        filename
        for filename in os.listdir(weights_dir)
        if filename.startswith(safe_model_tag) and filename.endswith(".pth")
    ]
    weight_files.sort()
    if not weight_files:
        raise FileNotFoundError(
            f"Nenhum .pth iniciado por '{safe_model_tag}' encontrado em: {weights_dir}"
        )

    return os.path.join(weights_dir, weight_files[0])


def _load_model(weight_path, model_name, version, num_classes, device):
    model = get_model(model_name, version=version, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _flatten_features(features):
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 4:
        features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=1)
    elif features.ndim == 3:
        features = features.mean(dim=1)
    return features.flatten(start_dim=1)


def _forward_embeddings(model, images):
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
        if hasattr(model, "forward_head"):
            try:
                features = model.forward_head(features, pre_logits=True)
            except TypeError:
                features = model.forward_head(features)
        return _flatten_features(features)

    logits = model(images)
    return _flatten_features(logits)


def _select_class_indices(dataset, class_name=None, class_label=None, max_real=None, max_fake=None):
    selected_real = []
    selected_fake = []

    for idx, metadata in enumerate(dataset.metadata):
        is_target_class = True
        if class_name is not None:
            is_target_class = metadata["class_name"].lower() == class_name.lower()
        if class_label is not None:
            is_target_class = is_target_class and metadata["label"] == class_label
        if not is_target_class:
            continue

        basename = os.path.basename(metadata["filepath"]).lower()
        is_fake = basename.startswith("sample")
        if is_fake:
            if max_fake is None or len(selected_fake) < max_fake:
                selected_fake.append(idx)
        else:
            if max_real is None or len(selected_real) < max_real:
                selected_real.append(idx)

    selected = selected_real + selected_fake
    domains = ["real"] * len(selected_real) + ["fake"] * len(selected_fake)
    return selected, domains


@torch.no_grad()
def _extract_embeddings(model, dataloader, device):
    embeddings = []
    for images, _ in tqdm(dataloader, desc="Extraindo embeddings", leave=False):
        images = images.to(device)
        batch_embeddings = _forward_embeddings(model, images).detach().cpu().numpy()
        embeddings.append(batch_embeddings)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def _plot_pca(coords, domains, output_path, title):
    colors = {"real": "#1f77b4", "fake": "#d62728"}
    markers = {"real": "o", "fake": "x"}

    plt.figure(figsize=(9, 7))
    for domain in ["real", "fake"]:
        mask = np.array(domains) == domain
        if not mask.any():
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[domain],
            marker=markers[domain],
            label=f"{domain} (n={mask.sum()})",
            alpha=0.75,
            edgecolors="none" if markers[domain] == "x" else "white",
            linewidths=0.4,
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def run_pca_visualization(
    weights_dir=None,
    weight_file=None,
    model_name="convnext_tiny",
    version="in12k_ft_in1k",
    class_name=None,
    class_label=None,
    fold_idx=0,
    batch_size=None,
    device=None,
    max_real=None,
    max_fake=None,
    output_prefix=None,
    standardize=True,
    config_path=None,
):
    config_dict = _load_runtime_config(config_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or config_dict["batch_size"]

    if class_name is None and class_label is None:
        raise ValueError("Informe --class-name ou --class-label.")

    _, eval_transform = build_transforms(config_dict["image_size"])
    train_dataset, _, _ = get_fold_datasets(
        fold_idx=fold_idx,
        transform_train=eval_transform,
        transform_eval=eval_transform,
        csv_metadata=config_dict["metadata_csv"],
        data_root=config_dict["data_dir"],
        split_csv_path=config_dict["split_csv_path"],
        num_folds=config_dict["num_folds"],
        random_state=config_dict["split_random_state"],
        filepath_column=config_dict["filepath_column"],
        label_column=config_dict["label_column"],
        class_name_column=config_dict["class_name_column"],
    )

    selected_indices, domains = _select_class_indices(
        train_dataset,
        class_name=class_name,
        class_label=class_label,
        max_real=max_real,
        max_fake=max_fake,
    )
    if len(selected_indices) < 2:
        raise ValueError("Foram encontradas menos de 2 imagens para essa selecao.")
    if len(set(domains)) < 2:
        raise ValueError("A selecao precisa ter imagens reais e falsas para comparar.")

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": config_dict["num_workers"],
        "pin_memory": config_dict["pin_memory"] and str(device).startswith("cuda"),
    }
    if config_dict["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = config_dict["persistent_workers"]
        if config_dict["prefetch_factor"] is not None:
            loader_kwargs["prefetch_factor"] = config_dict["prefetch_factor"]

    dataloader = DataLoader(Subset(train_dataset, selected_indices), **loader_kwargs)
    weight_path = _resolve_weight_path(weights_dir, weight_file, model_name, version, fold_idx)
    model = _load_model(weight_path, model_name, version, config_dict["num_classes"], device)

    embeddings = _extract_embeddings(model, dataloader, device)
    pca_input = StandardScaler().fit_transform(embeddings) if standardize else embeddings
    coords = PCA(n_components=2, random_state=42).fit_transform(pca_input)

    os.makedirs(config_dict["viz_dir"], exist_ok=True)
    os.makedirs(config_dict["results_dir"], exist_ok=True)
    model_tag, safe_model_tag = _safe_model_tag(model_name, version)
    class_token = class_name if class_name is not None else f"classe_{class_label}"
    class_token = str(class_token).replace(" ", "_")
    output_prefix = output_prefix or f"{safe_model_tag}_fold_{fold_idx}_{class_token}_pca_real_fake"
    plot_path = os.path.join(config_dict["viz_dir"], f"{output_prefix}.png")
    csv_path = os.path.join(config_dict["results_dir"], f"{output_prefix}.csv")

    selected_metadata = [train_dataset.metadata[idx] for idx in selected_indices]
    pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "domain": domains,
            "filepath": [metadata["filepath"] for metadata in selected_metadata],
            "label": [metadata["label"] for metadata in selected_metadata],
            "class_name": [metadata["class_name"] for metadata in selected_metadata],
            "sample_id": [metadata["sample_id"] for metadata in selected_metadata],
            "weight_path": weight_path,
        }
    ).to_csv(csv_path, index=False)

    title = f"PCA embeddings - {model_tag} | fold {fold_idx} | {class_token}"
    _plot_pca(coords, domains, plot_path, title)

    real_count = domains.count("real")
    fake_count = domains.count("fake")
    print(f"Peso carregado: {weight_path}")
    print(f"Amostras usadas: reais={real_count} | falsas={fake_count}")
    print(f"Figura salva em: {plot_path}")
    print(f"Coordenadas salvas em: {csv_path}")
    return plot_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualiza com PCA embeddings de imagens reais e falsas de uma classe do treino."
    )
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--weight-file", default=None)
    parser.add_argument("--model-name", default="convnext_tiny")
    parser.add_argument("--version", default="in12k_ft_in1k")
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pca_visualization(
        weights_dir=args.weights_dir,
        weight_file=args.weight_file,
        model_name=args.model_name,
        version=args.version,
        class_name=args.class_name,
        class_label=args.class_label,
        fold_idx=args.fold_idx,
        batch_size=args.batch_size,
        device=args.device,
        max_real=args.max_real,
        max_fake=args.max_fake,
        output_prefix=args.output_prefix,
        standardize=not args.no_standardize,
        config_path=args.config,
    )
