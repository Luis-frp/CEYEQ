import argparse
import json
import os
import re
import sys
import gc  # Para limpeza forçada de RAM
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.dataloader import get_fold_dataloaders
from models.generic_model import get_model
from utils.config import (
    _apply_environment,
    _apply_local_data_cache,
    _expand_config_values,
    _infer_dataset_classes,
    build_transforms,
)
from utils.train import build_confusion_matrix, collate_fn_skip_none, compute_per_class_metrics
from utils.visualizations import save_confusion_matrix


@dataclass
class WeightInfo:
    model_dir: str
    weight_path: str
    architecture_path: str
    fold: int
    model_name: str
    model_version: str | None
    num_classes: int
    dropout_rate: float
    image_size: list[int]

    @property
    def model_tag(self):
        return self.model_name if self.model_version is None else f"{self.model_name}.{self.model_version}"

    @property
    def model_key(self):
        return os.path.basename(self.model_dir)


def _load_runtime_config(config_path):
    if not os.path.exists(config_path):
        return {
            "image_size": [224, 224],
            "num_folds": 5,
            "split_random_state": 42,
            "num_classes": 3,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    _apply_environment(config_dict)
    config_dict = _expand_config_values(config_dict)

    for path_key in [
        "data_dir",
        "test_data_dir",
        "metadata_csv",
        "test_csv",
        "split_csv_path",
        "local_cache_dir",
    ]:
        if config_dict.get(path_key) and not os.path.isabs(config_dict[path_key]):
            config_dict[path_key] = os.path.join(PROJECT_ROOT, config_dict[path_key])

    config_dict.setdefault("image_size", [224, 224])
    config_dict.setdefault("num_folds", 5)
    config_dict.setdefault("split_random_state", 42)
    config_dict.setdefault("num_classes", 3)
    config_dict.setdefault("data_dir", os.path.join(PROJECT_ROOT, "data", "images"))
    config_dict.setdefault("test_data_dir", config_dict["data_dir"])
    config_dict.setdefault("metadata_csv", os.path.join(PROJECT_ROOT, "data", "new_train_eyeq_v2.csv"))
    config_dict.setdefault("test_csv", os.path.join(PROJECT_ROOT, "data", "teste_processado.csv"))
    config_dict.setdefault("split_csv_path", None)
    
    config_dict.setdefault("num_workers", 0) 
    config_dict.setdefault("pin_memory", False)
    config_dict.setdefault("persistent_workers", False)
    config_dict.setdefault("prefetch_factor", None)
    
    config_dict.setdefault("local_cache_dir", None)
    config_dict.setdefault("filepath_column", "image")
    config_dict.setdefault("label_column", "quality")
    config_dict.setdefault("class_name_column", None)
    config_dict.setdefault("dropout_rate", 0.3)

    _apply_local_data_cache(config_dict)
    _infer_dataset_classes(config_dict)
    return config_dict


def _safe_tag(value):
    return str(value).replace("/", "_")


def _extract_fold(path):
    match = re.search(r"_fold_(\d+)", os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def _split_model_tag(model_tag):
    if "." not in model_tag:
        return model_tag, None
    model_name, version = model_tag.split(".", 1)
    return model_name, version


def _load_architecture(architecture_path):
    if not architecture_path or not os.path.exists(architecture_path):
        return {}
    with open(architecture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _architecture_path_for_weight(weight_path):
    model_dir = os.path.dirname(os.path.dirname(weight_path))
    arch_dir = os.path.join(model_dir, "arquiteturas")
    stem = os.path.splitext(os.path.basename(weight_path))[0]
    return os.path.join(arch_dir, f"{stem}_arquitetura.json")


def _weight_sort_key(weight_info):
    return (weight_info.fold, weight_info.model_key.lower(), weight_info.model_tag.lower(), weight_info.weight_path)


def discover_weights(base_dir, default_config):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Pasta base nao encontrada: {base_dir}")

    weight_infos = []
    for root, _, files in os.walk(base_dir):
        if os.path.basename(root) != "weights":
            continue

        model_dir = os.path.dirname(root)
        for filename in files:
            if not filename.endswith(".pth"):
                continue

            weight_path = os.path.join(root, filename)
            architecture_path = _architecture_path_for_weight(weight_path)
            architecture = _load_architecture(architecture_path)
            stem = os.path.splitext(filename)[0]
            model_tag_from_file = re.sub(r"_fold_\d+$", "", stem)
            model_name, model_version = _split_model_tag(model_tag_from_file)

            weight_infos.append(
                WeightInfo(
                    model_dir=model_dir,
                    weight_path=weight_path,
                    architecture_path=architecture_path if os.path.exists(architecture_path) else "",
                    fold=_extract_fold(filename),
                    model_name=architecture.get("model_name", model_name),
                    model_version=architecture.get("model_version", model_version),
                    num_classes=int(architecture.get("num_classes", default_config["num_classes"])),
                    dropout_rate=float(architecture.get("dropout_rate", default_config.get("dropout_rate", 0.3))),
                    image_size=list(architecture.get("image_size", default_config["image_size"])),
                )
            )

    weight_infos.sort(key=_weight_sort_key)
    if not weight_infos:
        raise FileNotFoundError(f"Nenhum arquivo .pth encontrado em subpastas weights de: {base_dir}")
    return weight_infos


def _load_checkpoint(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def _build_model(weight_info, device):
    model = get_model(
        weight_info.model_name,
        version=weight_info.model_version,
        pretrained=False,
        num_classes=weight_info.num_classes,
        dropout_rate=weight_info.dropout_rate,
    )
    _load_checkpoint(model, weight_info.weight_path, device)
    return model.to(device).eval()


def _make_test_loader(config, image_size, batch_size, device):
    _, eval_transform = build_transforms(image_size)
    
    num_workers = int(config.get("num_workers", 0))
    pin_memory = config["pin_memory"] and device.type == "cuda"
    persistent_workers = config["persistent_workers"] and num_workers > 0
    prefetch_factor = config["prefetch_factor"] if num_workers > 0 else None

    _, _, test_loader = get_fold_dataloaders(
        fold_idx=0,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        transform_eval=eval_transform,
        csv_metadata=config.get("metadata_csv"),
        test_csv=config.get("test_csv"),
        data_root=config.get("data_dir"),
        test_data_root=config.get("test_data_dir"),
        split_csv_path=config.get("split_csv_path"),
        num_folds=config["num_folds"],
        random_state=config["split_random_state"],
        filepath_column=config["filepath_column"],
        label_column=config["label_column"],
        class_name_column=config["class_name_column"],
        collate_fn=collate_fn_skip_none,
    )
    return test_loader


@torch.no_grad()
def _predict_probabilities(model, dataloader, criterion, device, stage):
    targets = []
    probabilities = []
    running_loss = 0.0
    total_batches = 0

    for images, labels in tqdm(dataloader, desc=stage, leave=False):
        if images.numel() == 0:
            continue
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).cpu()

        targets.extend(labels.cpu().tolist())
        probabilities.extend(probs.tolist())
        running_loss += loss.item()
        total_batches += 1

    return targets, np.array(probabilities, dtype=np.float32), running_loss / max(total_batches, 1)


def _metrics_from_probabilities(targets, probabilities, loss=None):
    predictions = np.argmax(probabilities, axis=1).tolist()
    metrics = {
        "acc": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }
    if loss is not None:
        metrics["loss"] = loss
    return metrics, predictions


def _dataset_prediction_frame_with_std(dataset, targets, predictions, mean_probabilities, std_probabilities, class_names=None):
    """
    Gera o DataFrame incluindo as médias e os desvios padrões (incerteza) de predição por classe.
    """
    rows = []
    for idx, metadata in enumerate(dataset.metadata):
        image_path = metadata.get("image", metadata.get("filepath"))
        target = targets[idx]
        prediction = predictions[idx]
        target_name = metadata.get("class_name")
        if target_name is None:
            target_name = class_names[target] if class_names and target < len(class_names) else target

        row = {
            "sample_id": metadata.get("sample_id", image_path),
            "patient_id": metadata.get("patient_id"),
            "file": image_path,
            "target": target,
            "target_name": target_name,
            "prediction": prediction,
            "prediction_name": class_names[prediction] if class_names and prediction < len(class_names) else prediction,
            "predicted_mean_probability": float(mean_probabilities[idx][prediction]),
            "prediction_uncertainty_std": float(np.mean(std_probabilities[idx])), # Média de incerteza da amostra
        }
        
        # Detalhamento de Média e Desvio Padrão por classe
        for class_idx in range(mean_probabilities.shape[1]):
            class_label = class_names[class_idx] if class_names and class_idx < len(class_names) else class_idx
            row[f"prob_mean_{class_idx}_{class_label}"] = float(mean_probabilities[idx][class_idx])
            row[f"prob_std_{class_idx}_{class_label}"] = float(std_probabilities[idx][class_idx])
            
        rows.append(row)
    return pd.DataFrame(rows)


def _save_per_class_metrics(path, targets, predictions, labels, class_names, source):
    rows = compute_per_class_metrics(targets, predictions, labels=labels, class_names=class_names)
    for row in rows:
        row["source"] = source
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ensemble_by_fold(
    base_dir,
    config_path=os.path.join(PROJECT_ROOT, "config.yaml"),
    output_dir=None,
    batch_size=None,
    device=None,
):
    config = _load_runtime_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = batch_size or int(config.get("batch_size", 16))
    output_dir = output_dir or os.path.join(base_dir, "resultados_ensemble_por_fold")
    os.makedirs(output_dir, exist_ok=True)

    weight_infos = discover_weights(base_dir, config)
    
    # Agrupar pesos por fold
    weights_by_fold = defaultdict(list)
    for info in weight_infos:
        if info.fold != -1:
            weights_by_fold[info.fold].append(info)

    image_sizes = {tuple(info.image_size) for info in weight_infos}
    if len(image_sizes) > 1:
        raise ValueError(f"Os pesos usam image_size diferentes: {sorted(image_sizes)}")

    image_size = weight_infos[0].image_size
    test_loader = _make_test_loader(config, image_size, batch_size, device)
    criterion = nn.CrossEntropyLoss()
    class_labels = list(range(config["num_classes"]))
    class_names = config.get("class_names")

    print(f"Pesos encontrados agrupados em {len(weights_by_fold)} folds.")
    print(f"Dispositivo: {device} | batch_size={batch_size} | image_size={image_size}")
    print(f"Amostras no teste fixo: {len(test_loader.dataset)}")

    summary_rows = []
    final_targets = None

    for fold in sorted(weights_by_fold.keys()):
        fold_infos = weights_by_fold[fold]
        print(f"\n==================== INICIANDO ENSEMBLE FOLD {fold} ====================")
        print(f"Modelos neste fold: {[info.model_key for info in fold_infos]}")
        
        fold_probabilities = []
        
        for idx, info in enumerate(fold_infos, start=1):
            label = f"{info.model_key}/{info.model_tag}_fold_{info.fold}"
            print(f"\n  [{idx}/{len(fold_infos)}] Avaliando componente: {label}")
            
            model = _build_model(info, device)
            targets, probabilities, loss = _predict_probabilities(
                model,
                test_loader,
                criterion,
                device,
                stage=f"Teste - {label}",
            )
            metrics, _ = _metrics_from_probabilities(targets, probabilities, loss=loss)
            
            if final_targets is None:
                final_targets = targets
            elif final_targets != targets:
                raise ValueError("A ordem dos targets mudou entre avaliacoes. Ensemble abortado.")

            fold_probabilities.append(probabilities)
            
            summary_rows.append({
                "source": "individual",
                "fold_grupo": fold,
                "model_dir": info.model_key,
                "model_tag": info.model_tag,
                "fold_original": info.fold,
                **metrics,
            })
            print(f"  > Loss={metrics['loss']:.4f} | Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Conversão para array 3D: [Num_modelos, Num_amostras, Num_classes]
        stacked_probs = np.stack(fold_probabilities, axis=0)
        
        # --- Cálculo da Opção 1: Média e Desvio Padrão das Probabilidades por Imagem ---
        ensemble_fold_probabilities = np.mean(stacked_probs, axis=0)
        ensemble_fold_std = np.std(stacked_probs, axis=0)
        
        ensemble_fold_metrics, ensemble_fold_predictions = _metrics_from_probabilities(final_targets, ensemble_fold_probabilities)
        
        summary_rows.append({
            "source": f"ensemble_fold_{fold}",
            "fold_grupo": fold,
            "model_dir": f"ensemble_f{fold}",
            "model_tag": "media_probabilidades",
            "fold_original": fold,
            **ensemble_fold_metrics,
        })

        print(f"\n>>> RESULTADO DO ENSEMBLE FOLD {fold} <<<")
        print(f"Acc={ensemble_fold_metrics['acc']:.4f} | Precision={ensemble_fold_metrics['precision']:.4f} | Recall={ensemble_fold_metrics['recall']:.4f} | F1={ensemble_fold_metrics['f1']:.4f}")

        # Geração dos Outputs organizados
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Salvando as predições com as colunas agregadas de std (desvio padrão)
        _dataset_prediction_frame_with_std(
            test_loader.dataset, 
            final_targets, 
            ensemble_fold_predictions, 
            ensemble_fold_probabilities, 
            ensemble_fold_std,
            class_names=class_names
        ).to_csv(os.path.join(fold_output_dir, f"predicoes_ensemble_fold_{fold}.csv"), index=False)

        confusion_matrix = build_confusion_matrix(final_targets, ensemble_fold_predictions, labels=class_labels)
        save_confusion_matrix(
            confusion_matrix,
            os.path.join(fold_output_dir, f"matriz_confusao_ensemble_fold_{fold}.png"),
            title=f"Ensemble Fold {fold} - Matriz de Confusao",
            class_names=class_names,
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "resumo_geral_por_fold.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print("\n\n==================== RANKING DOS ENSEMBLES POR FOLD ====================")
    df_ensembles = df_summary[df_summary['source'].str.startswith('ensemble')].sort_values(by='f1', ascending=False)
    print(df_ensembles[['source', 'acc', 'f1', 'precision', 'recall']].to_string(index=False))
    print(f"\nResumo completo e estruturado salvo em: {summary_path}")

    return {
        "summary_csv": summary_path,
        "summary": df_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testa pesos e faz ensemble com desvio padrão agrupado por Fold."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Modelos", "Baseline"),
        help="Pasta base contendo subpastas de modelos.",
    )
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    test_ensemble_by_fold(
        base_dir=args.base_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
