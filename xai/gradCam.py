import os
import sys

# Adiciona a raiz do projeto ao path para conseguir importar helpers, models e utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from helpers.dataloader import get_fold_dataloaders
from models.generic_model import get_model
from utils.config import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from PIL import Image

LABEL_NAMES = {
    0: "Cancer",
    1: "Gerd",
    2: "Gerd Normal",
    3: "Polyp",
    4: "Normal Polyp",
    5: "Spot",
}

DEFAULT_IMAGE_SIZE = (224, 224)


def _denormalize_image(image_tensor):
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image_np = (image_np * std) + mean
    return np.clip(image_np, 0, 1)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("A biblioteca grad-cam não está instalada. Para usar esta funcionalidade, instale-a primeiro:")
    print("pip install grad-cam")

class BinaryTarget:
    def __init__(self, target_class):
        self.target_class = target_class
    def __call__(self, model_output):
        # model_output = [batch_size, 1] ou [batch_size]
        # Queremos maximizar o output (para gerar evidência da classe 1)
        # Ou minimizar o output (para gerar evidência da classe 0)
        out = model_output.squeeze()
        if self.target_class == 1:
            return out
        return -out

def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

def _infer_num_classes_from_state_dict(state_dict, default=1):
    for key in (
        "head.fc.weight",
        "head.weight",
        "fc.weight",
        "classifier.weight",
        "head.fc.bias",
        "head.bias",
        "fc.bias",
        "classifier.bias",
    ):
        weight = state_dict.get(key) if isinstance(state_dict, dict) else None
        if hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0])

    if isinstance(state_dict, dict):
        classifier_tensors = [
            tensor
            for key, tensor in state_dict.items()
            if hasattr(tensor, "shape")
            and len(tensor.shape) >= 1
            and any(token in key for token in ("head", "classifier", "fc"))
            and key.endswith((".weight", ".bias"))
        ]
        for tensor in sorted(classifier_tensors, key=lambda item: len(item.shape)):
            if int(tensor.shape[0]) > 1:
                return int(tensor.shape[0])
    return default

def _validate_weight_path(weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {weight_path}")

    valid_extensions = (".pt", ".pth", ".ckpt", ".bin")
    extension = os.path.splitext(weight_path)[1].lower()
    if extension not in valid_extensions:
        raise ValueError(
            "O caminho informado para os pesos nao parece ser um checkpoint "
            f"({', '.join(valid_extensions)}): {weight_path}"
        )

def generate_gradcam(
    model_name: str,
    version: str,
    weight_path: str,
    save_dir: str = "vizualizacoes",
    device: str = None,
    specific_image_path: str = None,
    specific_image_label: int = -1,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Carregando modelo {model_name}...")
    _validate_weight_path(weight_path)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    state_dict = _extract_state_dict(checkpoint)
    num_classes = _infer_num_classes_from_state_dict(state_dict, default=1)
    model = get_model(model_name, version=version, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # Define a camada alvo (target layer) típica do modelo MobileNetV3 ou outros no timm/torchvision
    try:
        # Preferir os blocos intermediários antes do pooling e camadas 1x1 finais
        if hasattr(model, 'blocks'):
            target_layers = [model.blocks[-1]] # Melhor para modelos timm (MobileNet, EfficientNet)
        elif hasattr(model, 'features'):
            target_layers = [model.features[-1]] # Padrão Torchvision
        elif hasattr(model, 'layer4'):
            target_layers = [model.layer4[-1]] # ResNets
        elif hasattr(model, 'conv_head'):
            target_layers = [model.conv_head] # Fallback para timm 
        else:
            target_layers = [list(model.children())[-2]] # Penúltima camada genérica
            
        print(f"Target layer identificada automaticamente: {target_layers[0].__class__.__name__}")
    except Exception as e:
        print(f"Atenção: Não foi possível identificar as target_layers automáticas. Detalhe do erro: {e}")
        return

    cam = GradCAM(model=model, target_layers=target_layers)
    os.makedirs(save_dir, exist_ok=True)
    
    # Verifica se o usuário forneceu uma imagem específica
    if specific_image_path and os.path.exists(specific_image_path):
        print(f"Extraindo mapa de ativação (Grad-CAM) para a imagem: {specific_image_path}")
        
        # Carregar a imagem e aplicar a mesma transformação do conjunto de testes
        pil_img = Image.open(specific_image_path).convert('RGB')
        _, transform = build_transforms(DEFAULT_IMAGE_SIZE)
        img_tensor = transform(pil_img).unsqueeze(0) # Adiciona dimensão do batch (1, C, H, W)
        
        test_samples = [(img_tensor, torch.tensor([specific_image_label]))]
        max_images = 1
    else:
        print("Carregando DataLoader de teste fixo...")
        _, _, test_loader = get_fold_dataloaders(fold_idx=0, batch_size=1, num_workers=0)
        test_samples = test_loader
        max_images = 10 # Limite de imagens para gerar o GradCAM
        print(f"Extraindo mapas de ativação (Grad-CAM) para as {max_images} primeiras imagens do teste...")

    count = 0
    for images, labels in test_samples:
        if count >= max_images:
            break
            
        input_tensor = images.to(device)
        
        # Faz uma predição primeiro para saber se a imagem caminha para classe 1 ou classe 0
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            if output.shape[-1] == 1:
                pred_score = torch.sigmoid(output).item()
                pred_class = 1 if pred_score >= 0.5 else 0
                pred_name = LABEL_NAMES.get(pred_class, f"Classe {pred_class}")
                pred_text = f"{pred_name} ({pred_class}) - Prob: {pred_score:.2f}"
                print(f"Predição: {pred_text}")
            else:
                probs = F.softmax(output, dim=1)
                pred_score, pred_class_tensor = torch.max(probs, dim=1)
                pred_class = int(pred_class_tensor.item())
                pred_name = LABEL_NAMES.get(pred_class, f"Classe {pred_class}")
                pred_text = f"{pred_name} ({pred_class}) - Prob: {pred_score.item():.2f}"
                prob_text = ", ".join(
                    f"{LABEL_NAMES.get(idx, f'Classe {idx}')}: {prob.item():.3f}"
                    for idx, prob in enumerate(probs[0])
                )
                print(f"Predição: {pred_text}")
                print(f"Probabilidades: {prob_text}")

        # Definir como target a própria predição (para visualizar o que motivou a decisão atual)
        targets = [BinaryTarget(pred_class)] if output.shape[-1] == 1 else [ClassifierOutputTarget(pred_class)]
        
        # Gerar o GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Desnormalizar a imagem apenas para exibir/sobrepor o Grad-CAM.
        img_np = _denormalize_image(input_tensor[0])
                
        # Sobrepor a cam na imagem original
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Salvar as figuras
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        
        lbl_val = labels[0].item()
        if lbl_val != -1:
            label_id = int(lbl_val)
            label_name = LABEL_NAMES.get(label_id, f"Classe {label_id}")
            label_text = f"Label: {label_name} ({label_id})"
        else:
            label_text = "Label: Desconhecido (Input manual)"
        axes[0].set_title(f"Original - {label_text}")
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        
        axes[1].set_title(f"GradCAM - Pred: {pred_text}")
        axes[1].axis('off')
        
        save_name = os.path.basename(specific_image_path) if specific_image_path and os.path.exists(specific_image_path) else f"gradcam_img_{count}.png"
        save_path = os.path.join(save_dir, f"grad-{save_name}" if specific_image_path else save_name)
        plt.savefig(save_path)
        plt.close()
        count += 1
        
    print(f"Mapas de ativação salvos na pasta: {save_dir}")

if __name__ == '__main__':
    ROOT_PATH = PROJECT_ROOT
    modelo_exemplo = "convnext_tiny"
    versao_exemplo = "in12k_ft_in1k"
    
    peso_relativo = os.path.join(
        "results",
        "weights",
        "conv",
        "convnext_tiny.in12k_ft_in1k_fold_3.pth",
    )
    pesos_exemplo = os.path.join(ROOT_PATH, peso_relativo)
    save_dir = os.path.join(ROOT_PATH, "vizualizacoes", "gradcam")
    
    # Exemplo: Especifique o caminho de uma imagem aqui, se desejar testar numa imagem específica
    # imagem_especifica = r'C:\Users\luisf\OneDrive\Desktop\Qualificação\CGLGIT\data\images\Cancer\004 (3).jpg' 

    specific_image_path = r'C:\Users\luisf\OneDrive\Desktop\Qualificação\CGLGIT\data\images\Spot\0brgf07.jpg'
    specific_image_label = 5  # Use -1 para desconhecido, ou 0-5 conforme LABEL_NAMES.
    if os.path.exists(pesos_exemplo):
        generate_gradcam(
            modelo_exemplo,
            versao_exemplo,
            pesos_exemplo,
            save_dir,
            specific_image_path=specific_image_path,
            specific_image_label=specific_image_label,
        )

    else:
        print(f"O caminho do peso não foi encontrado: {pesos_exemplo}")
