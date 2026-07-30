import os


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def save_training_curves(history, output_path, title=None):
    import matplotlib.pyplot as plt

    _ensure_parent_dir(output_path)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

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
    axes[1].set_title("Acuracia / F1")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Valor")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history.get("train_auc", []), label="Treino AUC")
    axes[2].plot(epochs, history.get("val_auc", []), label="Validacao AUC")
    axes[2].plot(epochs, history.get("train_kappa", []), label="Treino Kappa", linestyle="--")
    axes[2].plot(epochs, history.get("val_kappa", []), label="Validacao Kappa", linestyle="--")
    axes[2].set_title("AUC / Kappa")
    axes[2].set_xlabel("Epoca")
    axes[2].set_ylabel("Valor")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(targets, probabilities, output_path, class_names=None, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    _ensure_parent_dir(output_path)

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    num_classes = probabilities.shape[1]
    labels = list(range(num_classes))
    names = class_names if class_names is not None else [str(label) for label in labels]

    targets_binarized = label_binarize(targets, classes=labels)
    if num_classes == 2:
        targets_binarized = np.hstack([1 - targets_binarized, targets_binarized])

    fig, ax = plt.subplots(figsize=(6, 5))
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(mean_fpr)
    valid_curves = 0

    for class_idx in range(num_classes):
        if targets_binarized[:, class_idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(targets_binarized[:, class_idx], probabilities[:, class_idx])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{names[class_idx]} (AUC={class_auc:.3f})")
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        valid_curves += 1

    if valid_curves > 0:
        mean_tpr /= valid_curves
        macro_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, linestyle="--", color="black", label=f"Media macro (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title or "Curva ROC")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.3)

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
