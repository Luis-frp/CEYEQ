import argparse
import os
import sys
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.folder_dataset import FolderImageDataset
from utils.config import PROJECT_ROOT, _expand_config_values, _apply_environment, build_model, build_transforms


def _load_checkpoint(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def _load_config(config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    _apply_environment(config_dict)
    config_dict = _expand_config_values(config_dict)
    config_dict.setdefault("image_size", [224, 224])
    config_dict.setdefault("num_classes", 3)
    config_dict.setdefault("model_name", "resnet50")
    config_dict.setdefault("model_version", None)
    config_dict.setdefault("dropout_rate", 0.3)
    config_dict.setdefault("class_names", None)
    config_dict.setdefault("save_dir", os.path.join(PROJECT_ROOT, "weights"))
    return config_dict


@torch.no_grad()
def predict_folder(model, dataloader, device):
    model.eval()
    rows = []

    for images, metadata in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = torch.argmax(probabilities, dim=1).tolist()

        for idx, pred_class in enumerate(predictions):
            probs = probabilities[idx].tolist()
            rows.append(
                {
                    "file": metadata["relative_path"][idx],
                    "full_path": metadata["image_path"][idx],
                    "prediction": pred_class,
                    "predicted_probability": probs[pred_class],
                    **{f"prob_{class_idx}": prob for class_idx, prob in enumerate(probs)},
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_folder_ensemble(models, dataloader, device):
    for model in models:
        model.eval()

    rows = []

    for images, metadata in dataloader:
        images = images.to(device)
        batch_probs = None

        for model in models:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            batch_probs = probs if batch_probs is None else batch_probs + probs

        batch_probs = (batch_probs / len(models)).cpu()
        predictions = torch.argmax(batch_probs, dim=1).tolist()

        for idx, pred_class in enumerate(predictions):
            probs = batch_probs[idx].tolist()
            rows.append(
                {
                    "file": metadata["relative_path"][idx],
                    "full_path": metadata["image_path"][idx],
                    "prediction": pred_class,
                    "predicted_probability": probs[pred_class],
                    **{f"prob_{class_idx}": prob for class_idx, prob in enumerate(probs)},
                }
            )

    return pd.DataFrame(rows)


def _find_weight_files(weights_dir, model_name, version):
    model_tag = model_name if version is None else f"{model_name}.{version}"
    safe_model_tag = model_tag.replace("/", "_")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Pasta de pesos nao encontrada: {weights_dir}")
    weight_files = [
        os.path.join(weights_dir, filename)
        for filename in os.listdir(weights_dir)
        if filename.startswith(safe_model_tag) and filename.endswith(".pth")
    ]
    weight_files.sort()
    return weight_files


def _resolve_usable_class(args, class_names):
    if args.usable_class_name:
        if not class_names:
            raise ValueError(
                "Foi informado --usable-class-name, mas o config nao tem class_names. "
                "Use --usable-class ou defina class_names no config.yaml."
            )
        normalized = [str(name).strip().lower() for name in class_names]
        target = str(args.usable_class_name).strip().lower()
        if target not in normalized:
            raise ValueError(
                f"Classe '{args.usable_class_name}' nao encontrada em class_names={class_names}."
            )
        return normalized.index(target)
    return args.usable_class


def _copy_images_by_flag(df, output_dir, flag_column, source_root):
    copied_paths = []
    for _, row in df[df[flag_column]].iterrows():
        src = row["full_path"]
        rel = os.path.relpath(src, source_root)
        dst = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied_paths.append(dst)
    return copied_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Faz predicao em uma pasta de imagens e salva um CSV.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--weights", type=str, help="Caminho de um arquivo .pth do modelo treinado.")
    parser.add_argument("--weights-dir", type=str, help="Pasta com os .pth dos folds para fazer ensemble.")
    parser.add_argument("--input-dir", type=str, required=True, help="Pasta com as imagens para inferencia.")
    parser.add_argument("--output-csv", type=str, required=True, help="Arquivo CSV de saida.")
    parser.add_argument("--output-dir", type=str, required=True, help="Pasta de saida para copiar as imagens usaveis.")
    parser.add_argument("--output-dir-rejected", type=str, help="Pasta para copiar as imagens nao usaveis.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--usable-class", type=int, default=1, help="Indice da classe considerada 'usavel'.")
    parser.add_argument("--usable-class-name", type=str, help="Nome da classe considerada 'usavel'.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = build_transforms(config["image_size"])

    dataset = FolderImageDataset(args.input_dir, transform=eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.weights_dir:
        weight_files = _find_weight_files(args.weights_dir, config["model_name"], config["model_version"])
        if not weight_files:
            raise FileNotFoundError(f"Nenhum .pth encontrado em {args.weights_dir}.")
        models = [
            _load_checkpoint(
                build_model(
                    device=device,
                    model_name=config["model_name"],
                    model_version=config["model_version"],
                    pretrained=False,
                    num_classes=config["num_classes"],
                    dropout_rate=config["dropout_rate"],
                ),
                weight_path,
                device,
            )
            for weight_path in weight_files
        ]
        df = predict_folder_ensemble(models, dataloader, device)
    else:
        if not args.weights:
            raise ValueError("Informe --weights ou --weights-dir.")
        model = build_model(
            device=device,
            model_name=config["model_name"],
            model_version=config["model_version"],
            pretrained=False,
            num_classes=config["num_classes"],
            dropout_rate=config["dropout_rate"],
        )
        _load_checkpoint(model, args.weights, device)
        df = predict_folder(model, dataloader, device)

    class_names = config.get("class_names")
    usable_class_idx = _resolve_usable_class(args, class_names)
    if class_names:
        df["prediction_name"] = df["prediction"].map(lambda idx: class_names[idx])
        df["usable_class_name"] = class_names[usable_class_idx]
    else:
        df["usable_class_name"] = usable_class_idx
    df["usable_predicted"] = df["prediction"] == usable_class_idx
    df["usable_predicted_name"] = df["usable_predicted"].map({True: "usavel", False: "nao_usavel"})

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"CSV salvo em: {args.output_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    copied_usable = _copy_images_by_flag(df, args.output_dir, "usable_predicted", args.input_dir)
    print(f"Imagens usaveis copiadas: {len(copied_usable)}")
    print(f"Pasta de saida usaveis: {args.output_dir}")

    if args.output_dir_rejected:
        os.makedirs(args.output_dir_rejected, exist_ok=True)
        copied_rejected = []
        for _, row in df[~df["usable_predicted"]].iterrows():
            src = row["full_path"]
            rel = os.path.relpath(src, args.input_dir)
            dst = os.path.join(args.output_dir_rejected, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied_rejected.append(dst)
        print(f"Imagens nao usaveis copiadas: {len(copied_rejected)}")
        print(f"Pasta de saida nao usaveis: {args.output_dir_rejected}")


if __name__ == "__main__":
    main()
