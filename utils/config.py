import argparse
import json
import os
import re
import shutil
import sys
import yaml
import pandas as pd
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

from helpers.dataloader import get_fold_dataloaders
from helpers.metadata import read_metadata_csv
from models.generic_model import get_model, list_available_models, list_versions
from utils.train import (
    build_confusion_matrix,
    compute_per_class_metrics,
    evaluate,
    fit_fold,
    predict_classes,
    collate_fn_skip_none,
)
from utils.visualizations import save_confusion_matrix, save_training_curves


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento com StratifiedGroupKFold e YAML config.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config.yaml"), help="Path para o arquivo de configuracao YAML.")
    
    # Pode manter algumas flags CLI que sobrescrevam o YAML ou usar apenas o YAML.
    # Aqui estamos focando em ler diretamente o YAML
    args_cli = parser.parse_args()
    
    config_path = args_cli.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuracao {config_path} nao encontrado.")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    _apply_environment(config_dict)
    config_dict = _expand_config_values(config_dict)

    # Converter caminhos relativos para absolutos caso precise.
    for dir_key in ["save_dir", "viz_dir", "arch_dir", "results_dir", "data_dir", "metadata_csv", "test_csv", "split_csv_path", "local_cache_dir"]:
        if dir_key in config_dict and config_dict[dir_key] and not os.path.isabs(config_dict[dir_key]):
            config_dict[dir_key] = os.path.join(PROJECT_ROOT, config_dict[dir_key])
            
    # Garantir default de fold_idx para iterar em todos se null
    if "fold_idx" not in config_dict or config_dict["fold_idx"] is None:
        config_dict["fold_idx"] = None
        
    config_dict.setdefault("interactive", True)
    config_dict.setdefault("model_name", "resnet50")
    config_dict.setdefault("model_version", None)
    config_dict.setdefault("dropout_rate", 0.3)
    config_dict.setdefault("data_augmentation", True)
    config_dict.setdefault("results_dir", os.path.join(PROJECT_ROOT, "resultados"))
    config_dict.setdefault("data_dir", os.path.join(PROJECT_ROOT, "data", "images"))
    config_dict.setdefault("metadata_csv", os.path.join(PROJECT_ROOT, "data", "new_train_eyeq_v2.csv"))
    config_dict.setdefault("test_csv", os.path.join(PROJECT_ROOT, "data", "teste_processado.csv"))
    config_dict.setdefault("split_csv_path", None)
    config_dict.setdefault("split_random_state", 42)
    config_dict.setdefault("pin_memory", True)
    config_dict.setdefault("persistent_workers", True)
    config_dict.setdefault("prefetch_factor", 2)
    config_dict.setdefault("onecycle_pct_start", 0.1)
    config_dict.setdefault("onecycle_anneal_strategy", "cos")
    config_dict.setdefault("onecycle_div_factor", 100.0)
    config_dict.setdefault("onecycle_final_div_factor", 10000.0)
    config_dict.setdefault("local_cache_dir", None)
    config_dict.setdefault("filepath_column", "image")
    config_dict.setdefault("label_column", "quality")
    config_dict.setdefault("class_name_column", None)
    _apply_local_data_cache(config_dict)
    _infer_dataset_classes(config_dict)

    return SimpleNamespace(**config_dict)


def _apply_environment(config_dict):
    environment = config_dict.get("environment") or config_dict.get("env") or {}
    for key, value in environment.items():
        if value is not None and not os.environ.get(str(key)):
            os.environ[str(key)] = str(value)


def _expand_config_values(value):
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(_expand_env_with_defaults(value))
    return value


def _expand_env_with_defaults(value):
    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def replace(match):
        env_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value != "":
            return env_value
        return default if default is not None else match.group(0)

    return os.path.expandvars(pattern.sub(replace, value))


def _infer_dataset_classes(config_dict):
    metadata_csv = config_dict.get("metadata_csv")
    label_column = config_dict.get("label_column", "quality")
    class_name_column = config_dict.get("class_name_column")
    if not metadata_csv or not os.path.exists(metadata_csv):
        return

    df = read_metadata_csv(metadata_csv)
    if label_column not in df.columns and "quality" in df.columns:
        label_column = "quality"
    if label_column not in df.columns:
        return

    labels = sorted(int(label) for label in df[label_column].dropna().unique().tolist())
    inferred_num_classes = max(labels) + 1 if labels else 0
    current_num_classes = config_dict.get("num_classes")
    if current_num_classes in (None, "auto") or int(current_num_classes) <= 1 < inferred_num_classes:
        config_dict["num_classes"] = inferred_num_classes

    if class_name_column in df.columns:
        class_map = (
            df[[label_column, class_name_column]]
            .drop_duplicates()
            .sort_values(label_column)
            .set_index(label_column)[class_name_column]
            .to_dict()
        )
        config_dict["class_names"] = [str(class_map.get(label, label)) for label in range(config_dict["num_classes"])]


def _copy_if_missing(source_path, target_path):
    if not source_path or not os.path.exists(source_path):
        return source_path
    if os.path.abspath(source_path) == os.path.abspath(target_path):
        return target_path

    if os.path.isdir(source_path):
        if os.path.isdir(target_path) and os.listdir(target_path):
            return target_path
        os.makedirs(target_path, exist_ok=True)
        print(f"Copiando imagens para cache local: {source_path} -> {target_path}")
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        return target_path

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    if not os.path.exists(target_path):
        print(f"Copiando metadata para cache local: {source_path} -> {target_path}")
        shutil.copy2(source_path, target_path)
    return target_path


def _apply_local_data_cache(config_dict):
    local_cache_dir = config_dict.get("local_cache_dir")
    if not local_cache_dir:
        return

    os.makedirs(local_cache_dir, exist_ok=True)
    config_dict["data_dir"] = _copy_if_missing(
        config_dict.get("data_dir"),
        os.path.join(local_cache_dir, "images"),
    )
    config_dict["metadata_csv"] = _copy_if_missing(
        config_dict.get("metadata_csv"),
        os.path.join(local_cache_dir, "metadata.csv"),
    )


def _is_exit_choice(value):
    return value.lower() in ["0", "q", "sair", "exit"]


def _prompt_model_selection(default_model_name="resnet50", default_model_version=None):
    print("\n--- Selecao de modelo ---")
    print("Voce pode informar um filtro do timm, por exemplo: resnet*, efficientnet*, convnext*, vit*")
    print("DICA: Digite '0' ou 'q' a qualquer momento para sair do programa sem erro.")
    default_family = re.match(r"[A-Za-z]+", default_model_name or "")
    filtro_default = f"{default_family.group(0)}*" if default_family else "resnet*"
    filtro = input(f"Filtro [Enter para usar {filtro_default}]: ").strip()
    
    if _is_exit_choice(filtro):
        print("Saindo...")
        sys.exit(0)
    
    filtro = filtro or filtro_default

    modelos = list_available_models(filtro=filtro, apenas_pretrained=True)
    if not modelos:
        raise ValueError(f"Nenhum modelo encontrado para o filtro '{filtro}'.")

    max_items = min(120, len(modelos))
    print(f"\nModelos encontrados ({len(modelos)}). Mostrando os primeiros {max_items}:")
    for idx, model_name in enumerate(modelos[:max_items], start=1):
        print(f"[{idx}] {model_name}")

    escolha = input(f"\nEscolha o indice ou digite o nome do modelo [Enter para {default_model_name}]: ").strip()
    
    if _is_exit_choice(escolha):
        print("Saindo...")
        sys.exit(0)
        
    escolha = escolha or default_model_name
    
    if escolha.isdigit():
        indice = int(escolha)
        if indice < 1 or indice > max_items:
            raise ValueError(f"Indice invalido. Escolha entre 1 e {max_items}.")
        model_name = modelos[indice - 1]
    else:
        model_name = escolha

    versoes = list_versions(model_name)
    model_version = default_model_version
    if versoes:
        print(f"\nVersoes disponiveis para '{model_name}':")
        for idx, version in enumerate(versoes[:10], start=1):
            print(f"[{idx}] {version}")
    else:
        print(f"\nNenhuma versao listada automaticamente para '{model_name}'.")

    version_default = default_model_version or "padrao do timm"
    version_choice = input(f"Escolha a versao por indice/nome [Enter para {version_default}]: ").strip()

    if _is_exit_choice(version_choice):
        print("Saindo...")
        sys.exit(0)

    if version_choice:
        if version_choice.isdigit() and versoes:
            indice = int(version_choice)
            if indice < 1 or indice > min(10, len(versoes)):
                raise ValueError("Indice de versao invalido.")
            model_version = versoes[indice - 1]
        else:
            model_version = version_choice

    return model_name, model_version


def _prompt_image_size(default_image_size):
    default_size = default_image_size[0] if isinstance(default_image_size, (list, tuple)) else default_image_size
    raw_value = input(f"\nInforme o tamanho da imagem [Enter para {default_size}] ou 'q' para sair: ").strip()
    
    if _is_exit_choice(raw_value):
        print("Saindo...")
        sys.exit(0)
        
    raw_value = raw_value or str(default_size)
    image_size = int(raw_value)
    return [image_size, image_size]


def apply_interactive_menu(args):
    print("Menu de treinamento")
    model_name, model_version = _prompt_model_selection(
        default_model_name=args.model_name,
        default_model_version=args.model_version,
    )
    image_size = _prompt_image_size(args.image_size)

    args.model_name = model_name
    args.model_version = model_version
    args.image_size = image_size

    print(f"\nModelo selecionado: {args.model_name}{'.' + args.model_version if args.model_version else ''}")
    print(f"Tamanho da imagem: {args.image_size[0]}x{args.image_size[1]}")
    return args


def build_transforms(image_size):
    image_size = tuple(image_size)
    normalization = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        normalization,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalization,
    ])

    return train_transform, eval_transform


def build_model(device, model_name, model_version=None, pretrained=True, num_classes=6, dropout_rate=0.0):
    model = get_model(
        model_name=model_name,
        version=model_version,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    return model.to(device)


def _safe_model_tag(args):
    model_tag = args.model_name if args.model_version is None else f"{args.model_name}.{args.model_version}"
    return model_tag, model_tag.replace("/", "_")


def _dataset_prediction_frame(dataset, targets, predictions, probabilities, fold_idx, split_name, class_names=None):
    rows = []
    for idx, metadata in enumerate(dataset.metadata):
        image_path = metadata.get("image", metadata.get("filepath"))
        target_name = metadata.get("class_name")
        if target_name is None:
            target_name = class_names[targets[idx]] if class_names and targets[idx] < len(class_names) else targets[idx]
        row = {
            "fold": fold_idx,
            "split": split_name,
            "sample_id": metadata.get("sample_id", image_path),
            "patient_id": metadata.get("patient_id"),
            "file": image_path,
            "target": targets[idx],
            "target_name": target_name,
            "prediction": predictions[idx],
            "prediction_name": class_names[predictions[idx]] if class_names and predictions[idx] < len(class_names) else predictions[idx],
            "predicted_probability": probabilities[idx][predictions[idx]],
        }
        for class_idx, probability in enumerate(probabilities[idx]):
            class_label = class_names[class_idx] if class_names and class_idx < len(class_names) else class_idx
            row[f"prob_{class_idx}_{class_label}"] = probability
        rows.append(row)
    return pd.DataFrame(rows)


def _save_fold_csvs(args, safe_model_tag, fold_idx, fold_result, val_frame, test_frame):
    fold_dir = os.path.join(args.results_dir, f"{safe_model_tag}_fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    metrics_path = os.path.join(fold_dir, "metricas_validacao_teste.csv")
    predictions_path = os.path.join(fold_dir, "predicoes_validacao_teste.csv")

    metrics_rows = [
        {
            "fold": fold_idx,
            "split": "validacao",
            "loss": fold_result["best_val_loss"],
            "acc": fold_result["best_val_acc"],
            "precision": fold_result["best_val_precision"],
            "recall": fold_result["best_val_recall"],
            "f1": fold_result["best_val_f1"],
            "best_epoch": fold_result["best_epoch"],
        },
        {
            "fold": fold_idx,
            "split": "teste_fixo",
            "loss": fold_result["test_loss"],
            "acc": fold_result["test_acc"],
            "precision": fold_result["test_precision"],
            "recall": fold_result["test_recall"],
            "f1": fold_result["test_f1"],
            "best_epoch": fold_result["best_epoch"],
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.concat([val_frame, test_frame], ignore_index=True).to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(args.image_size)
    if not args.data_augmentation:
        train_transform = None
    fold_results = []
    fold_confusion_matrices = []
    per_class_metric_rows = []
    all_test_probabilities = []
    final_test_targets = None
    class_labels = list(range(args.num_classes))
    class_names = getattr(args, "class_names", None)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.arch_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.fold_idx is None:
        folds_to_run = range(args.num_folds)
    else:
        if args.fold_idx < 0 or args.fold_idx >= args.num_folds:
            raise ValueError(f"--fold-idx deve estar entre 0 e {args.num_folds - 1}.")
        folds_to_run = [args.fold_idx]

    print(f"Treinando em: {device}")
    print(f"CSV de metadados: {args.metadata_csv}")
    print(f"Diretorio das imagens: {args.data_dir}")
    print(f"CSV de teste: {getattr(args, 'test_csv', None)}")
    print(f"Numero de classes: {args.num_classes}")

    for fold_idx in folds_to_run:
        print(f"\n===== Fold {fold_idx + 1}/{args.num_folds} =====")
        model_tag, safe_model_tag = _safe_model_tag(args)

        train_loader, val_loader, test_loader = get_fold_dataloaders(
            fold_idx=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device.type == "cuda",
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            transform_train=train_transform,
            transform_eval=eval_transform,
            csv_metadata=args.metadata_csv,
            test_csv=getattr(args, "test_csv", None),
            data_root=args.data_dir,
            split_csv_path=args.split_csv_path,
            num_folds=args.num_folds,
            random_state=args.split_random_state,
            filepath_column=args.filepath_column,
            label_column=args.label_column,
            class_name_column=args.class_name_column,
            collate_fn=collate_fn_skip_none,
        )
        print(
            f"Amostras no fold {fold_idx}: "
            f"treino={len(train_loader.dataset)} | "
            f"validacao={len(val_loader.dataset)} | "
            f"teste={len(test_loader.dataset)}"
        )

        model = build_model(
            device=device,
            model_name=args.model_name,
            model_version=args.model_version,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.onecycle_pct_start,
            anneal_strategy=args.onecycle_anneal_strategy,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
        )

        model, history, best_metrics = fit_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.early_stop_patience,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            class_labels=class_labels,
            class_names=class_names,
        )

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
        val_targets, val_predictions, val_probabilities = predict_classes(
            model=model,
            dataloader=val_loader,
            device=device,
            stage="Val Predict",
            return_probabilities=True,
        )
        test_targets, test_predictions, test_probabilities = predict_classes(
            model=model,
            dataloader=test_loader,
            device=device,
            stage="Test Predict CSV",
            return_probabilities=True,
        )
        if final_test_targets is None:
            final_test_targets = test_targets
        all_test_probabilities.append(test_probabilities)
        confusion_matrix = build_confusion_matrix(test_targets, test_predictions, labels=class_labels)
        fold_confusion_matrices.append(confusion_matrix)
        for row in compute_per_class_metrics(
            test_targets,
            test_predictions,
            labels=class_labels,
            class_names=class_names,
        ):
            row["fold"] = fold_idx
            row["split"] = "teste_fixo"
            per_class_metric_rows.append(row)

        fold_result = {
            "fold": fold_idx,
            "best_epoch": best_metrics["epoch"] + 1,
            "best_val_loss": best_metrics["val_loss"],
            "best_val_acc": best_metrics["val_acc"],
            "best_val_precision": best_metrics["val_precision"],
            "best_val_recall": best_metrics["val_recall"],
            "best_val_f1": best_metrics["val_f1"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        }

        weight_path = os.path.join(args.save_dir, f"{safe_model_tag}_fold_{fold_idx}.pth")
        curve_path = os.path.join(args.viz_dir, f"{safe_model_tag}_fold_{fold_idx}_curvas.png")
        confusion_matrix_path = os.path.join(args.viz_dir, f"{safe_model_tag}_fold_{fold_idx}_matriz_confusao.png")
        architecture_path = os.path.join(args.arch_dir, f"{safe_model_tag}_fold_{fold_idx}_arquitetura.json")
        torch.save(model.state_dict(), weight_path)
        save_training_curves(history, curve_path, title=f"{model_tag} - Fold {fold_idx}")
        save_confusion_matrix(
            confusion_matrix,
            confusion_matrix_path,
            title=f"{model_tag} - Fold {fold_idx}",
            class_names=class_names,
        )
        
        architecture_data = {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "full_model_id": model_tag,
            "pretrained": args.pretrained,
            "num_classes": args.num_classes,
            "dropout_rate": args.dropout_rate,
            "class_names": class_names,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "scheduler": "OneCycleLR",
            "onecycle_pct_start": args.onecycle_pct_start,
            "onecycle_anneal_strategy": args.onecycle_anneal_strategy,
            "onecycle_div_factor": args.onecycle_div_factor,
            "onecycle_final_div_factor": args.onecycle_final_div_factor,
            "num_epochs_requested": args.num_epochs,
            "early_stop_patience": args.early_stop_patience,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "best_epoch": best_metrics["epoch"] + 1,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "model_repr": str(model),
        }
        with open(architecture_path, "w") as architecture_file:
            json.dump(architecture_data, architecture_file, indent=2)

        val_frame = _dataset_prediction_frame(
            val_loader.dataset,
            val_targets,
            val_predictions,
            val_probabilities,
            fold_idx,
            "validacao",
            class_names=class_names,
        )
        test_frame = _dataset_prediction_frame(
            test_loader.dataset,
            test_targets,
            test_predictions,
            test_probabilities,
            fold_idx,
            "teste_fixo",
            class_names=class_names,
        )
        metrics_csv_path, predictions_csv_path = _save_fold_csvs(
            args=args,
            safe_model_tag=safe_model_tag,
            fold_idx=fold_idx,
            fold_result=fold_result,
            val_frame=val_frame,
            test_frame=test_frame,
        )
        fold_result["weight_path"] = weight_path
        fold_result["curve_path"] = curve_path
        fold_result["confusion_matrix_path"] = confusion_matrix_path
        fold_result["architecture_path"] = architecture_path
        fold_result["metrics_csv_path"] = metrics_csv_path
        fold_result["predictions_csv_path"] = predictions_csv_path
        fold_results.append(fold_result)

        print(
            f"Fold {fold_idx} finalizado | "
            f"best_epoch={fold_result['best_epoch']} | "
            f"best_val_loss={fold_result['best_val_loss']:.4f} | "
            f"best_val_acc={fold_result['best_val_acc']:.4f} | "
            f"best_val_precision={fold_result['best_val_precision']:.4f} | "
            f"best_val_recall={fold_result['best_val_recall']:.4f} | "
            f"best_val_f1={fold_result['best_val_f1']:.4f} | "
            f"test_loss={fold_result['test_loss']:.4f} | "
            f"test_acc={fold_result['test_acc']:.4f} | "
            f"test_precision={fold_result['test_precision']:.4f} | "
            f"test_recall={fold_result['test_recall']:.4f} | "
            f"test_f1={fold_result['test_f1']:.4f} | "
            f"weights={weight_path} | "
            f"curvas={curve_path} | "
            f"matriz={confusion_matrix_path} | "
            f"arquitetura={architecture_path} | "
            f"metricas_csv={metrics_csv_path} | "
            f"predicoes_csv={predictions_csv_path}"
        )

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    mean_test_loss = sum(result["test_loss"] for result in fold_results) / len(fold_results)
    mean_test_acc = sum(result["test_acc"] for result in fold_results) / len(fold_results)
    mean_test_precision = sum(result["test_precision"] for result in fold_results) / len(fold_results)
    mean_test_recall = sum(result["test_recall"] for result in fold_results) / len(fold_results)
    mean_test_f1 = sum(result["test_f1"] for result in fold_results) / len(fold_results)
    summary_csv_path = os.path.join(args.results_dir, "resumo_folds.csv")
    pd.DataFrame(fold_results).to_csv(summary_csv_path, index=False)

    per_class_metrics_path = os.path.join(args.results_dir, f"{safe_model_tag}_metricas_por_classe_folds.csv")
    per_class_summary_path = os.path.join(args.results_dir, f"{safe_model_tag}_metricas_por_classe_media_desvio.csv")
    mean_confusion_matrix_path = os.path.join(args.viz_dir, f"{safe_model_tag}_matriz_confusao_media.png")
    mean_confusion_matrix_csv_path = os.path.join(args.results_dir, f"{safe_model_tag}_matriz_confusao_media.csv")

    per_class_summary = pd.DataFrame()
    if per_class_metric_rows:
        per_class_df = pd.DataFrame(per_class_metric_rows)
        per_class_df.to_csv(per_class_metrics_path, index=False)
        std_ddof0 = lambda values: values.std(ddof=0)
        per_class_summary = (
            per_class_df
            .groupby(["class", "class_name"], as_index=False)
            .agg(
                precision_mean=("precision", "mean"),
                precision_std=("precision", std_ddof0),
                recall_mean=("recall", "mean"),
                recall_std=("recall", std_ddof0),
                f1_mean=("f1", "mean"),
                f1_std=("f1", std_ddof0),
                support_mean=("support", "mean"),
                support_std=("support", std_ddof0),
            )
            .fillna(0.0)
        )
        per_class_summary.to_csv(per_class_summary_path, index=False)

    if fold_confusion_matrices:
        import numpy as np

        mean_confusion_matrix = np.mean(np.array(fold_confusion_matrices, dtype=float), axis=0)
        pd.DataFrame(
            mean_confusion_matrix,
            index=class_names if class_names else class_labels,
            columns=class_names if class_names else class_labels,
        ).to_csv(mean_confusion_matrix_csv_path)
        save_confusion_matrix(
            mean_confusion_matrix,
            mean_confusion_matrix_path,
            title=f"{model_tag} - Matriz de Confusao Media",
            class_names=class_names,
            value_format=".2f",
        )

    # Calculo do Ensemble (Maioria dos Votos no Teste Fixo) usando sklearn.metrics
    if len(all_test_probabilities) > 0 and final_test_targets is not None:
        import numpy as np

        probs_array = np.array(all_test_probabilities)
        ensemble_preds = np.argmax(np.mean(probs_array, axis=0), axis=1).tolist()
        
        ensemble_acc = accuracy_score(final_test_targets, ensemble_preds)
        ensemble_precision = precision_score(final_test_targets, ensemble_preds, average="macro", zero_division=0)
        ensemble_recall = recall_score(final_test_targets, ensemble_preds, average="macro", zero_division=0)
        ensemble_f1 = f1_score(final_test_targets, ensemble_preds, average="macro", zero_division=0)
    else:
        ensemble_acc = ensemble_precision = ensemble_recall = ensemble_f1 = 0.0

    print("\n===== Resumo Final =====")
    for result in fold_results:
        print(
            f"Fold {result['fold']} | "
            f"best_epoch={result['best_epoch']} | "
            f"best_val_loss={result['best_val_loss']:.4f} | "
            f"best_val_acc={result['best_val_acc']:.4f} | "
            f"best_val_precision={result['best_val_precision']:.4f} | "
            f"best_val_recall={result['best_val_recall']:.4f} | "
            f"best_val_f1={result['best_val_f1']:.4f} | "
            f"test_loss={result['test_loss']:.4f} | "
            f"test_acc={result['test_acc']:.4f} | "
            f"test_precision={result['test_precision']:.4f} | "
            f"test_recall={result['test_recall']:.4f} | "
            f"test_f1={result['test_f1']:.4f} | "
            f"weights={result['weight_path']} | "
            f"curvas={result['curve_path']} | "
            f"matriz={result['confusion_matrix_path']} | "
            f"arquitetura={result['architecture_path']} | "
            f"metricas_csv={result['metrics_csv_path']} | "
            f"predicoes_csv={result['predictions_csv_path']}"
        )

    print(f"Media Aritmetica Isolada test_loss: {mean_test_loss:.4f}")
    print(f"Media Aritmetica Isolada test_acc: {mean_test_acc:.4f}")
    print(f"Media Aritmetica Isolada test_precision: {mean_test_precision:.4f}")
    print(f"Media Aritmetica Isolada test_recall: {mean_test_recall:.4f}")
    print(f"Media Aritmetica Isolada test_f1: {mean_test_f1:.4f}")
    print(f"Resumo dos folds salvo em: {summary_csv_path}")
    if not per_class_summary.empty:
        print("\n===== Media e Desvio Padrao por Classe (Teste Fixo) =====")
        for _, row in per_class_summary.iterrows():
            print(
                f"Classe {row['class']} ({row['class_name']}) | "
                f"precision={row['precision_mean']:.4f} +/- {row['precision_std']:.4f} | "
                f"recall={row['recall_mean']:.4f} +/- {row['recall_std']:.4f} | "
                f"f1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | "
                f"support={row['support_mean']:.2f} +/- {row['support_std']:.2f}"
            )
        print(f"Metricas por classe dos folds salvas em: {per_class_metrics_path}")
        print(f"Resumo por classe salvo em: {per_class_summary_path}")
    if fold_confusion_matrices:
        print(f"Matriz de confusao media salva em: {mean_confusion_matrix_path}")
        print(f"CSV da matriz de confusao media salvo em: {mean_confusion_matrix_csv_path}")
    
    print("\n===== Resultado do Ensemble (Votacao Real via Sklearn) =====")
    print(f"Ensemble test_acc: {ensemble_acc:.4f}")
    print(f"Ensemble test_precision: {ensemble_precision:.4f}")
    print(f"Ensemble test_recall: {ensemble_recall:.4f}")
    print(f"Ensemble test_f1: {ensemble_f1:.4f}")
