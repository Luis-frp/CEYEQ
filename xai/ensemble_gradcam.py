import os
import sys
import gc
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Garante que o caminho do projeto está no sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.dataloader import get_fold_dataloaders
from models.generic_model import get_model
from utils.config import IMAGENET_MEAN, IMAGENET_STD, build_transforms

# Importando helpers do seu script de testes original
from seu_script_de_testes_original import (
    _load_runtime_config,
    discover_weights,
    _build_model,
    _make_test_loader
)

# Tenta importar a biblioteca pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Por favor, instale a biblioteca do GradCAM antes de rodar:")
    print("pip install grad-cam")
    sys.exit(1)

# Mapeamento de classes para EyeQ (Ajuste se o ID das suas classes for diferente)
EYEQ_CLASSES = {
    0: "Good",
    1: "Usable",
    2: "Reject"
}


def _denormalize_image(image_tensor):
    """Reverte a normalização do ImageNet para exibição em RGB."""
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image_np = (image_np * std) + mean
    return np.clip(image_np, 0, 1)


def _get_target_layer(model):
    """Detecta automaticamente a última camada convolucional útil para o GradCAM."""
    if hasattr(model, 'blocks'):
        return [model.blocks[-1]]       # Modelos timm (MobileNetV3, EfficientNet)
    elif hasattr(model, 'features'):
        return [model.features[-1]]     # Padrão PyTorch (VGG, AlexNet)
    elif hasattr(model, 'layer4'):
        return [model.layer4[-1]]       # ResNets
    elif hasattr(model, 'conv_head'):
        return [model.conv_head]        # Fallback timm
    else:
        return [list(model.children())[-2]]  # Penúltima camada genérica


def find_representative_images(dataloader, num_classes=3):
    """
    Busca no DataLoader uma imagem real de exemplo para cada uma das classes.
    Retorna um dicionário: { class_id: (image_tensor, label_tensor) }
    """
    samples = {}
    print("Buscando amostras representativas das classes no DataLoader...")
    for images, labels in dataloader:
        for i in range(images.size(0)):
            label = int(labels[i].item())
            if label not in samples and label < num_classes:
                # Mantém a dimensão de batch [1, C, H, W]
                samples[label] = (images[i:i+1], labels[i:i+1])
            
            if len(samples) == num_classes:
                return samples
    return samples


def generate_ensemble_gradcam(
    base_dir,
    fold_to_visualize=0,
    config_path=os.path.join(PROJECT_ROOT, "config.yaml"),
    save_dir="vizualizacoes/ensemble_gradcam",
    device=None
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = _load_runtime_config(config_path)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Carrega dados e descobre os pesos salvos
    weight_infos = discover_weights(base_dir, config)
    fold_weights = [info for info in weight_infos if info.fold == fold_to_visualize]

    if not fold_weights:
        raise ValueError(f"Nenhum peso encontrado para o fold {fold_to_visualize}")

    image_size = fold_weights[0].image_size
    test_loader = _make_test_loader(config, image_size, batch_size=4, device=device)
    
    # 2. Seleciona 1 imagem de exemplo de cada classe (Good, Usable, Reject)
    representative_samples = find_representative_images(test_loader, num_classes=3)
    
    # 3. Inicializa e carrega todos os modelos do fold selecionado na memória de forma temporária
    models_loaded = []
    for info in fold_weights:
        print(f"Carregando modelo: {info.model_key} (Fold {info.fold})")
        model = _build_model(info, device)
        models_loaded.append((info.model_key, model))

    # 4. Processa cada uma das classes
    for class_id, (img_tensor, label_tensor) in representative_samples.items():
        class_name = EYEQ_CLASSES.get(class_id, f"Classe_{class_id}")
        print(f"\n--- Gerando Grad-CAM para a classe: {class_name} ---")
        
        img_input = img_tensor.to(device)
        img_rgb = _denormalize_image(img_tensor[0])
        
        individual_cams = []
        model_names = []
        predictions_text = []

        # Executa o Grad-CAM em cada modelo individual
        for model_name, model in models_loaded:
            target_layers = _get_target_layer(model)
            cam_extractor = GradCAM(model=model, target_layers=target_layers)
            
            # Executa predição rápida para exibir na legenda
            with torch.no_grad():
                output = model(img_input)
                probs = F.softmax(output, dim=1)[0]
                pred_class = int(torch.argmax(probs).item())
                pred_prob = float(probs[pred_class])
                predictions_text.append(f"{model_name}\nPred: {EYEQ_CLASSES.get(pred_class, pred_class)} ({pred_prob:.2%})")

            # Alvo é a classe real do exemplo
            targets = [ClassifierOutputTarget(class_id)]
            
            # Gera o mapa de calor de ativação (valores normalizados de 0 a 1)
            grayscale_cam = cam_extractor(input_tensor=img_input, targets=targets)[0, :]
            individual_cams.append(grayscale_cam)
            model_names.append(model_name)

        # 5. Calcula a Média dos Mapas de Ativação (Ensemble Grad-CAM)
        ensemble_cam = np.mean(np.stack(individual_cams, axis=0), axis=0)
        
        # 6. Monta a plotagem comparativa (Lado a Lado)
        # Colunas: Imagem Original | Model_1 CAM | Model_2 CAM | ... | Ensemble CAM
        num_cols = len(models_loaded) + 2
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5))
        
        # Plot 1: Imagem Original
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original\nClasse Real: {class_name}", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Plots intermediários: Grad-CAMs individuais
        for idx, grayscale_cam in enumerate(individual_cams):
            vis = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
            axes[idx + 1].imshow(vis)
            axes[idx + 1].set_title(predictions_text[idx], fontsize=9)
            axes[idx + 1].axis('off')
            
        # Último Plot: Grad-CAM do Ensemble (Média)
        ensemble_vis = show_cam_on_image(img_rgb, ensemble_cam, use_rgb=True)
        axes[-1].imshow(ensemble_vis)
        axes[-1].set_title("ENSEMBLE\n(Média dos Mapas)", fontsize=10, fontweight='bold', color='darkblue')
        axes[-1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"comparativo_gradcam_{class_name.lower()}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Salvo comparativo da classe {class_name} em: {save_path}")

    # Limpeza de memória
    for _, model in models_loaded:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # Configure aqui o diretório onde estão as pastas "weights" dos seus modelos treinados
    BASE_DIR_EXEMPLO = os.path.join(PROJECT_ROOT, "Modelos", "Baseline")
    
    generate_ensemble_gradcam(
        base_dir=BASE_DIR_EXEMPLO,
        fold_to_visualize=0,  # Defina qual fold deseja analisar
        save_dir="vizualizacoes/gradcam_ensemble"
    )
