import os


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def save_training_curves(history, output_path, title=None):
    import matplotlib.pyplot as plt

    _ensure_parent_dir(output_path)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="Treino")
    axes[0].plot(epochs, history.get("val_loss", []), label="Validacao")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.get("train_f1", []), label="Treino F1")
    axes[1].plot(epochs, history.get("val_f1", []), label="Validacao F1")
    axes[1].plot(epochs, history.get("train_acc", []), label="Treino Acc", linestyle="--")
    axes[1].plot(epochs, history.get("val_acc", []), label="Validacao Acc", linestyle="--")
    axes[1].set_title("Metricas")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Valor")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(confusion_matrix, output_path, title=None, class_names=None, value_format=None):
    import matplotlib.pyplot as plt
    import numpy as np

    _ensure_parent_dir(output_path)

    matrix = np.asarray(confusion_matrix)
    labels = class_names if class_names is not None else [str(idx) for idx in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title or "Matriz de Confusao")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = format(value, value_format) if value_format else str(value)
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
