import os
import sys
import numpy as np

# Adiciona a raiz do projeto ao path para conseguir importar helpers, models e utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helpers.dataloader import get_fold_dataloaders
from models.generic_model import get_model
from utils.train import build_confusion_matrix, compute_per_class_metrics, evaluate, predict_classes
from utils.visualizations import save_confusion_matrix
from utils.config import PROJECT_ROOT, _apply_environment, _apply_local_data_cache, _expand_config_values, _infer_dataset_classes, build_transforms
import yaml


def _load_runtime_config():
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        return {"image_size": [224, 224], "num_folds": 5, "split_random_state": 42, "num_classes": 3}

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    _apply_environment(config_dict)
    config_dict = _expand_config_values(config_dict)

    for path_key in ["data_dir", "metadata_csv", "test_csv", "split_csv_path", "results_dir", "viz_dir", "local_cache_dir"]:
        if path_key in config_dict and config_dict[path_key] and not os.path.isabs(config_dict[path_key]):
            config_dict[path_key] = os.path.join(PROJECT_ROOT, config_dict[path_key])

    config_dict.setdefault("image_size", [224, 224])
    config_dict.setdefault("num_folds", 5)
    config_dict.setdefault("split_random_state", 42)
    config_dict.setdefault("num_classes", 3)
    config_dict.setdefault("data_dir", os.path.join(PROJECT_ROOT, "data", "images"))
    config_dict.setdefault("metadata_csv", os.path.join(PROJECT_ROOT, "data", "new_train_eyeq_v2.csv"))
    config_dict.setdefault("test_csv", os.path.join(PROJECT_ROOT, "data", "teste_processado.csv"))
    config_dict.setdefault("split_csv_path", None)
    config_dict.setdefault("results_dir", os.path.join(PROJECT_ROOT, "resultados"))
    config_dict.setdefault("viz_dir", os.path.join(PROJECT_ROOT, "vizualizacoes"))
    config_dict.setdefault("num_workers", 4)
    config_dict.setdefault("pin_memory", True)
    config_dict.setdefault("persistent_workers", True)
    config_dict.setdefault("prefetch_factor", 2)
    config_dict.setdefault("local_cache_dir", None)
    config_dict.setdefault("filepath_column", "image")
    config_dict.setdefault("label_column", "quality")
    config_dict.setdefault("class_name_column", None)
    _apply_local_data_cache(config_dict)
    _infer_dataset_classes(config_dict)
    return config_dict

def test_all_folds(weights_dir: str, model_name: str, version: str = None, batch_size: int = 32, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_tag = model_name if version is None else f"{model_name}.{version}"
    safe_model_tag = model_tag.replace("/", "_")

    # Lê as configurações salvas (ex: tamanho da imagem)
    config_dict = _load_runtime_config()
    image_size = config_dict["image_size"]

    _, eval_transform = build_transforms(image_size)

    print("Carregando DataLoader de teste fixo...")
    # Qualquer fold (ex: 0) retornará o mesmo test_loader que contém o split fixo
    _, _, test_loader = get_fold_dataloaders(
        fold_idx=0, 
        batch_size=batch_size, 
        num_workers=config_dict["num_workers"],
        pin_memory=config_dict["pin_memory"] and str(device).startswith("cuda"),
        persistent_workers=config_dict["persistent_workers"],
        prefetch_factor=config_dict["prefetch_factor"],
        transform_eval=eval_transform,
        csv_metadata=config_dict.get("metadata_csv"),
        test_csv=config_dict.get("test_csv"),
        data_root=config_dict.get("data_dir"),
        split_csv_path=config_dict.get("split_csv_path"),
        num_folds=config_dict["num_folds"],
        random_state=config_dict["split_random_state"],
        filepath_column=config_dict["filepath_column"],
        label_column=config_dict["label_column"],
        class_name_column=config_dict["class_name_column"],
    )
    
    criterion = nn.CrossEntropyLoss()
    
    metrics_list = {
        "loss": [],
        "acc": [],
        "f1": [],
        "precision": [],
        "recall": []
    }
    class_labels = list(range(config_dict["num_classes"]))
    class_names = config_dict.get("class_names")
    fold_confusion_matrices = []
    per_class_metric_rows = []
    
    print(f"\n--- Iniciando Validação para: {model_tag} ---")
    
    # Procura os arquivos de pesos na pasta
    weight_files = [f for f in os.listdir(weights_dir) if f.startswith(safe_model_tag) and f.endswith(".pth")]
    weight_files.sort()
    
    if not weight_files:
        print(f"Nenhum arquivo de peso encontrado para '{safe_model_tag}' em '{weights_dir}'.")
        return
        
    for w_file in weight_files:
        weight_path = os.path.join(weights_dir, w_file)
        
        print(f"\n[+] Avaliando: {w_file}")
        model = get_model(model_name, version=version, pretrained=False, num_classes=config_dict["num_classes"])
        
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        
        # Coletar predições para compor a média de métricas isoladas
        targets, predictions = predict_classes(model, test_loader, device, stage=f"Predicao - {w_file}")
        
        # Calcular os scores via Sklearn (mesma rotina do evaluate)
        loss_val = evaluate(model, test_loader, criterion, device, stage="Metrica")["loss"]
        
        # Ou diretamente do pred que acabamos de fazer
        acc_val = accuracy_score(targets, predictions)
        prec_val = precision_score(targets, predictions, average="macro", zero_division=0)
        rec_val = recall_score(targets, predictions, average="macro", zero_division=0)
        f1_val = f1_score(targets, predictions, average="macro", zero_division=0)
        confusion_matrix = build_confusion_matrix(targets, predictions, labels=class_labels)
        fold_confusion_matrices.append(confusion_matrix)
        fold_label = os.path.splitext(w_file)[0]
        for row in compute_per_class_metrics(
            targets,
            predictions,
            labels=class_labels,
            class_names=class_names,
        ):
            row["fold"] = fold_label
            row["split"] = "teste_fixo"
            per_class_metric_rows.append(row)
        
        metrics_list["loss"].append(loss_val)
        metrics_list["acc"].append(acc_val)
        metrics_list["precision"].append(prec_val)
        metrics_list["recall"].append(rec_val)
        metrics_list["f1"].append(f1_val)
        
        print(f"Loss: {loss_val:.4f} | Acc: {acc_val:.4f} | Prec: {prec_val:.4f} | Rec: {rec_val:.4f} | F1: {f1_val:.4f}")
        
    print("\n==================================")
    print("   RESUMO ESTATÍSTICO DOS FOLDS   ")
    print("==================================")
    for m in ["loss", "acc", "precision", "recall", "f1"]:
        arr = np.array(metrics_list[m])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        print(f"Média {m.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs(config_dict["results_dir"], exist_ok=True)
    os.makedirs(config_dict["viz_dir"], exist_ok=True)

    if per_class_metric_rows:
        import pandas as pd

        per_class_df = pd.DataFrame(per_class_metric_rows)
        per_class_metrics_path = os.path.join(config_dict["results_dir"], f"{safe_model_tag}_teste_metricas_por_classe_folds.csv")
        per_class_summary_path = os.path.join(config_dict["results_dir"], f"{safe_model_tag}_teste_metricas_por_classe_media_desvio.csv")
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

        print("\n===== MEDIA E DESVIO PADRAO POR CLASSE =====")
        for _, row in per_class_summary.iterrows():
            print(
                f"Classe {row['class']} ({row['class_name']}) | "
                f"Precision: {row['precision_mean']:.4f} +/- {row['precision_std']:.4f} | "
                f"Recall: {row['recall_mean']:.4f} +/- {row['recall_std']:.4f} | "
                f"F1: {row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | "
                f"Support: {row['support_mean']:.2f} +/- {row['support_std']:.2f}"
            )
        print(f"Metricas por classe dos folds salvas em: {per_class_metrics_path}")
        print(f"Resumo por classe salvo em: {per_class_summary_path}")

    if fold_confusion_matrices:
        import pandas as pd

        mean_confusion_matrix = np.mean(np.array(fold_confusion_matrices, dtype=float), axis=0)
        mean_confusion_matrix_path = os.path.join(config_dict["viz_dir"], f"{safe_model_tag}_teste_matriz_confusao_media.png")
        mean_confusion_matrix_csv_path = os.path.join(config_dict["results_dir"], f"{safe_model_tag}_teste_matriz_confusao_media.csv")
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
        print(f"Matriz de confusao media salva em: {mean_confusion_matrix_path}")
        print(f"CSV da matriz de confusao media salvo em: {mean_confusion_matrix_csv_path}")


if __name__ == '__main__':
    ROOT_PATH = os.path.join(PROJECT_ROOT, r"C:\Users\luisf\OneDrive\Desktop\Qualificação\CGLGIT\results\com_sintetic_cancer\weights")
    
    # Substitua pelo nome do seu modelo e versão caso queira testar outro
    modelo_exemplo = "convnext_tiny"
    versao_exemplo = "in12k_ft_in1k"
    
    if os.path.exists(ROOT_PATH):
        test_all_folds(ROOT_PATH, modelo_exemplo, versao_exemplo)
    else:
        print(f"A pasta weights não existe ainda em: {ROOT_PATH}")
