import sys
import timm
from typing import List, Optional
import json
from datetime import datetime
from pathlib import Path

def list_available_models(filtro: Optional[str] = None, apenas_pretrained: bool = True) -> List[str]:
    """
    Lista modelos disponíveis no timm. Use filtro como 'resnet*' para reduzir resultados.
    """
    modelos = timm.list_models(filter=filtro, pretrained=apenas_pretrained)
    return sorted(modelos)

def list_versions(model_name: str) -> List[str]:
    """
    Lista as versões/variantes de pesos pré-treinados para um modelo (tags como 'in1k', 'a1h', etc).
    Retorna apenas as tags; o nome completo da variante no timm é 'model.tag' (ex: 'resnet50.a1_in1k').
    """
    variantes: List[str] = []

    # Tentativa via API moderna do timm
    get_cfg = getattr(timm, "get_pretrained_cfg", None)
    if callable(get_cfg):
        try:
            cfg = get_cfg(model_name)
            # cfg.pretrained_cfgs é um dict {tag: cfg}
            tags = getattr(cfg, "pretrained_cfgs", None)
            if isinstance(tags, dict):
                variantes = list(tags.keys())
            else:
                # Alguns modelos só têm um cfg com tag única
                tag_unica = getattr(cfg, "tag", None)
                if tag_unica:
                    variantes = [tag_unica]
        except Exception:
            pass

    # Fallback: instanciar sem pesos e inspecionar atributos
    if not variantes:
        try:
            m = timm.create_model(model_name, pretrained=False)
            if hasattr(m, "pretrained_cfgs") and isinstance(m.pretrained_cfgs, dict):
                variantes = list(m.pretrained_cfgs.keys())
            elif hasattr(m, "pretrained_cfg") and isinstance(m.pretrained_cfg, dict):
                tag_unica = m.pretrained_cfg.get("tag")
                if tag_unica:
                    variantes = [tag_unica]
            elif hasattr(m, "default_cfg"):
                # Timms antigos: apenas config padrão
                variantes = []
        except Exception:
            pass

    # Ordena e remove vazios
    return sorted([v for v in variantes if v])

def get_model(
    model_name: str,
    version: Optional[str] = None,
    pretrained: bool = True,
    num_classes: int = 1,
    dropout_rate: float = 0.0,
):
    """
    Retorna o modelo com base no nome e, opcionalmente, na versão (tag) de pesos pré-treinados.
    Se 'version' for fornecida, usa a sintaxe 'model.version' (ex: 'resnet50.a1_in1k').
    """
    try:
        model_id = f"{model_name}.{version}" if version else model_name
        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes,
        }
        if dropout_rate and dropout_rate > 0:
            create_kwargs["drop_rate"] = dropout_rate

        try:
            model = timm.create_model(model_id, **create_kwargs)
        except TypeError as exc:
            if "drop_rate" not in create_kwargs:
                raise
            create_kwargs.pop("drop_rate")
            print(f"Aviso: '{model_id}' nao aceitou drop_rate={dropout_rate}. Criando sem dropout configuravel.")
            model = timm.create_model(model_id, **create_kwargs)

        return model
    except KeyError:
        raise ValueError(f"Modelo '{model_name}' não é suportado pelo timm.")
    except Exception as e:
        raise ValueError(f"Falha ao carregar '{model_name}'"
                         f"{'.' + version if version else ''}: {e}")

def save_model_selection(model_name: str, version: Optional[str], num_classes: int, output_path: Optional[str] = None):
    if output_path is None:
        # Salva por padrão na raiz do projeto RQIA
        base_dir = Path(__file__).resolve().parents[1]
        output_path = base_dir / "selected_model.json"
    else:
        output_path = Path(output_path)

    data = {
        "model_name": model_name,
        "version": version,
        "full_id": f"{model_name}{'.' + version if version else ''}",
        "num_classes": num_classes,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"Seleção salva em {output_path}")
    except Exception as e:
        print(f"Falha ao salvar JSON: {e}")

def menu(preselected: Optional[str] = None):
    """
        Só pra controlar melhor a seleção de modelos via terminal.
        Se preselected for informado (ex.: 'resnet50.a1_in1k' ou 'volo_d1_384'),
        o modelo é criado diretamente sem interação.
    """
    if preselected:
        pre = preselected.strip()
        if pre.lower().endswith('.pth'):
            pre = pre[:-4]
        # separa usando o primeiro ponto: model_name . tag
        if "." in pre:
            model_name, version = pre.split(".", 1)
        else:
            model_name, version = pre, None
        print(f"Carregando modelo predefinido: {model_name}{'.' + version if version else ''}")
        modelo = get_model(model_name, version=version, pretrained=True, num_classes=3, dropout_rate=0.3)
        save_model_selection(model_name, version, 3)
        return modelo

    print("Opcional: informe um filtro (ex: 'resnet*', 'vit*', 'convnext*'). Pressione Enter para listar todos.")
    filtro = input("Filtro: ").strip() or None

    modelos = list_available_models(filtro=filtro, apenas_pretrained=True)
    if not modelos:
        print("Nenhum modelo encontrado com esse filtro.")
        sys.exit(1)

    print(f"\nModelos disponíveis ({len(modelos)}):")
    for i, m in enumerate(modelos, start=1):
        print(f"[{i}] {m}")

    escolha = input("\nSelecione o índice ou digite o nome do modelo: ").strip()
    if escolha.isdigit():
        idx = int(escolha)
        if idx < 1 or idx > len(modelos):
            print("Índice inválido.")
            sys.exit(1)
        modelo_escolhido = modelos[idx - 1]
    else:
        modelo_escolhido = escolha
        if modelo_escolhido not in modelos:
            print("Modelo não encontrado na lista. Tentando assim mesmo...")

    versoes = list_versions(modelo_escolhido)
    if versoes:
        print(f"\nVersões disponíveis para '{modelo_escolhido}' ({len(versoes)}):")
        for i, v in enumerate(versoes, start=1):
            print(f"[{i}] {v}  -> {modelo_escolhido}.{v}")
        vsel = input("\nSelecione a versão (índice, nome da tag) ou pressione Enter para usar padrão: ").strip()
        if vsel:
            if vsel.isdigit():
                vidx = int(vsel)
                if 1 <= vidx <= len(versoes):
                    versao = versoes[vidx - 1]
                else:
                    print("Índice de versão inválido. Usando padrão.")
                    versao = None
            else:
                versao = vsel if vsel in versoes else None
                if versao is None:
                    print("Tag não reconhecida. Usando padrão.")
        else:
            versao = None
    else:
        print(f"\nNenhuma versão anotada para '{modelo_escolhido}'. Usando pesos padrão (se disponíveis).")
        versao = None

    print("\nCarregando o modelo, aguarde...")
    modelo = get_model(modelo_escolhido, version=versao, pretrained=True, num_classes=3, dropout_rate=0.3)
    full_name = f"{modelo_escolhido}{'.' + versao if versao else ''}"
    print(f"Modelo '{full_name}' carregado com sucesso -> 3 classes fixo!")
    # salva seleção em JSON
    save_model_selection(modelo_escolhido, versao, 3)

    return modelo
