# CEYEQ

Pipeline de classificação de qualidade de imagens de fundo de olho (retinografias), adaptado a partir do **EYEQ Dataset** e datasets afins (FQS/FIQS). Treina modelos do [timm](https://github.com/huggingface/pytorch-image-models) com validação cruzada estratificada por paciente, gera métricas, curvas ROC, matrizes de confusão, mapas de explicabilidade (Grad-CAM) e suporta ensemble de folds/modelos.

## Visão geral

O objetivo é classificar imagens de retina em classes de qualidade (ex.: usável / rejeitada), com foco em:

- Treinamento com **K-Fold estratificado por grupo** (`StratifiedGroupKFold`), agrupando imagens pelo mesmo paciente para evitar *leakage* entre treino/validação/teste.
- Suporte a qualquer arquitetura disponível no `timm` (ResNet, ConvNeXt, EfficientNet, ViT, etc.), com seleção interativa ou via `config.yaml`.
- **OneCycleLR** + early stopping durante o treino.
- Avaliação com métricas por classe (precisão, recall, F1, kappa, AUC) e **ensemble por votação/probabilidade média** entre folds.
- Explicabilidade via **Grad-CAM** (individual e por ensemble).
- Inferência em pastas de imagens não rotuladas, com separação automática em usável/rejeitada.
- Visualização de espaço de features via **PCA**.

## Estrutura do projeto

```
.
├── main.py                    # Ponto de entrada do treinamento
├── config.yaml                # Configuração central (modelo, dados, hiperparâmetros)
├── ensemble.sh                # Exemplo de chamada para avaliação em ensemble
├── models/
│   └── generic_model.py       # Wrapper sobre o timm (list/get de modelos e versões)
├── helpers/
│   ├── dataset.py             # EyeQDataset: leitura de imagens a partir de um CSV de metadados
│   ├── folder_dataset.py      # Dataset para inferência em pasta "solta" de imagens
│   ├── dataloader.py          # Split K-Fold por paciente e construção dos DataLoaders
│   └── metadata.py            # Leitura de CSVs de metadados
├── utils/
│   ├── config.py              # Parsing de config.yaml, montagem de transforms/modelo, loop de treino
│   ├── train.py               # Loop de treino/avaliação, métricas, matriz de confusão
│   ├── teste_ensemble.py      # Avaliação em ensemble de múltiplos folds/modelos treinados
│   ├── infer_folder.py        # Inferência em uma pasta de imagens (com ou sem ensemble)
│   ├── pca_vizu.py            # Projeção PCA das features extraídas pelo backbone
│   └── visualizations.py      # Curvas de treino, matriz de confusão, curva ROC
├── xai/
│   ├── gradCam.py             # Grad-CAM por classe para um único modelo/fold
│   └── ensemble_gradcam.py    # Grad-CAM agregado entre múltiplos modelos do ensemble
├── arquiteturas/              # JSON com metadados de arquitetura por fold (gerado no treino)
└── data/                      # CSVs de metadados (treino/teste)
```

> Diretórios como `weights/`, `resultados/`, `vizualizacoes/`, `Modelos/`, `FQS/`, `MQIS/` e os pesos `*.pth` não são versionados (ver `.gitignore`) — são gerados/baixados localmente.

## Requisitos

- Python 3.10+
- GPU com CUDA (opcional, mas recomendado — o código cai para CPU automaticamente se `torch.cuda.is_available()` for `False`)

Principais dependências (não há `requirements.txt` no repositório ainda; instale conforme necessário):

```bash
pip install torch torchvision timm pandas numpy scikit-learn matplotlib pillow pyyaml tqdm
```

## Configuração

Todo o treinamento é orientado pelo [`config.yaml`](config.yaml). Principais chaves:

| Chave | Descrição |
|---|---|
| `model_name`, `model_version` | Nome/versão do modelo no `timm` (ex.: `convnext_small`, `fb_in22k_ft_in1k`) |
| `num_epochs`, `batch_size`, `learning_rate` | Hiperparâmetros de treino |
| `image_size` | Tamanho de entrada `[H, W]` |
| `data_augmentation` | Ativa/desativa augmentations no treino |
| `num_folds`, `fold_idx`, `split_random_state` | Configuração do K-Fold (deixe `fold_idx: null` para treinar todos os folds) |
| `early_stop_patience` | Paciência do early stopping |
| `onecycle_*` | Parâmetros do scheduler `OneCycleLR` |
| `freeze_backbone_epochs` | Épocas iniciais com backbone congelado |
| `data_dir`, `metadata_csv` | Pasta de imagens e CSV de metadados de treino/validação |
| `test_data_dir`, `test_csv` | Pasta de imagens e CSV do conjunto de teste fixo |
| `filepath_column`, `label_column`, `class_name_column` | Nomes das colunas relevantes no CSV |
| `save_dir`, `viz_dir`, `arch_dir`, `results_dir` | Diretórios de saída (pesos, gráficos, arquitetura, métricas) |
| `interactive` | Se `true`, abre um menu no terminal para escolher modelo/versão/tamanho de imagem |

O CSV de metadados precisa ter, no mínimo, uma coluna com o caminho da imagem (`image` por padrão) e uma coluna com o rótulo numérico (`quality` por padrão). O número de classes é inferido automaticamente a partir dos rótulos presentes no CSV.

Variáveis de ambiente podem ser referenciadas no YAML com a sintaxe `${VAR:-default}` (ver `local_cache_dir` em `config.yaml`), útil para cachear imagens/CSVs localmente antes do treino.

## Uso

### Treinar

```bash
python main.py --config config.yaml
```

Se `interactive: true` no config (ou passado via menu), o script pergunta o filtro de modelo (`resnet*`, `convnext*`, `vit*`, ...), a versão pré-treinada e o tamanho da imagem antes de iniciar.

Para cada fold treinado, são gerados: pesos (`weights/*.pth`), curvas de treino, matriz de confusão, curvas ROC (validação e teste), Grad-CAM por classe, JSON de arquitetura e CSVs de métricas/predições. Ao final, é calculado um **ensemble por média de probabilidades** entre os folds, com métricas agregadas e resumo por classe (média ± desvio padrão).

### Avaliar um ensemble de modelos já treinados

```bash
python utils/teste_ensemble.py \
    --base-dir Modelos/Baseline/melhores \
    --output-dir teste_ensemble_results/FQS/Baseline \
    --batch-size 16 \
    --device cuda
```

(ver [`ensemble.sh`](ensemble.sh) para um exemplo pronto). `--base-dir` deve conter subpastas por modelo, cada uma com os pesos (`.pth`) e o JSON de arquitetura gerados no treino.

### Inferência em uma pasta de imagens (sem rótulos)

```bash
python utils/infer_folder.py \
    --config config.yaml \
    --weights-dir weights/meu_ensemble \
    --input-dir /caminho/para/imagens \
    --output-csv resultados/predicoes.csv \
    --output-dir /caminho/para/usaveis \
    --output-dir-rejected /caminho/para/rejeitadas
```

Aceita um único `--weights` (modelo/fold) ou `--weights-dir` (ensemble por média de probabilidades). As imagens são copiadas para as pastas de saída de acordo com a classe prevista (`--usable-class`).

### Grad-CAM

```bash
python xai/gradCam.py --config config.yaml            # por modelo/fold
python xai/ensemble_gradcam.py --config config.yaml   # agregado entre modelos do ensemble
```

### Visualização PCA das features

```bash
python utils/pca_vizu.py --config config.yaml --weights-dir weights/meu_ensemble --fold-idx 0
```

## Saídas geradas

- `weights/` — checkpoints por fold (`.pth`), incluindo histórico de treino e métricas.
- `vizualizacoes/` — curvas de treino, matrizes de confusão, curvas ROC e Grad-CAM (imagens `.png`).
- `arquiteturas/` — JSON com hiperparâmetros e metadados de cada modelo/fold treinado.
- `resultados/` — CSVs com métricas por fold, por classe, predições individuais e resumo do ensemble.
