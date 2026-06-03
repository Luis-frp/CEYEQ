import copy

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def _resolve_module(model, dotted_name):
    current = model
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def freeze_backbone_keep_head_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    head_module_names = (
        "classifier",
        "fc",
        "head",
        "head.fc",
        "heads",
        "heads.head",
        "last_linear",
    )

    unfrozen_params = 0
    for module_name in head_module_names:
        module = _resolve_module(model, module_name)
        if module is None:
            continue
        for parameter in module.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                unfrozen_params += parameter.numel()

    if unfrozen_params == 0:
        for name, parameter in model.named_parameters():
            if any(token in name for token in ("classifier", "fc", "head", "last_linear")):
                parameter.requires_grad = True
                unfrozen_params += parameter.numel()

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def unfreeze_all_layers(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def _move_batch_to_device(images, labels, device):
    return images.to(device), labels.to(device, dtype=torch.long)


def _forward_loss(model, criterion, images, labels):
    logits = model(images)
    loss = criterion(logits, labels)
    return logits, loss


def _compute_classification_metrics(targets, predictions):
    if len(targets) == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "acc": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, num_epochs=None):
    model.train()
    running_loss = 0.0
    total_batches = 0
    all_targets = []
    all_preds = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch + 1}/{num_epochs}" if epoch is not None and num_epochs is not None else "Train",
        leave=False,
    )

    for images, labels in progress_bar:
        images, labels = _move_batch_to_device(images, labels, device)

        optimizer.zero_grad()
        logits, loss = _forward_loss(model, criterion, images, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1).cpu()
        targets = labels.cpu()

        all_targets.extend(targets.tolist())
        all_preds.extend(preds.tolist())

        running_loss += loss.item()
        total_batches += 1
        metrics = _compute_classification_metrics(all_targets, all_preds)
        progress_bar.set_postfix(
            loss=f"{running_loss / total_batches:.4f}",
            acc=f"{metrics['acc']:.4f}",
            f1=f"{metrics['f1']:.4f}",
        )

    metrics = _compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(total_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, stage="Eval", epoch=None, num_epochs=None):
    model.eval()
    running_loss = 0.0
    total_batches = 0
    all_targets = []
    all_preds = []

    progress_bar = tqdm(
        dataloader,
        desc=f"{stage} Epoch {epoch + 1}/{num_epochs}" if epoch is not None and num_epochs is not None else stage,
        leave=False,
    )

    for images, labels in progress_bar:
        images, labels = _move_batch_to_device(images, labels, device)
        logits, loss = _forward_loss(model, criterion, images, labels)

        preds = torch.argmax(logits, dim=1).cpu()
        targets = labels.cpu()

        all_targets.extend(targets.tolist())
        all_preds.extend(preds.tolist())

        running_loss += loss.item()
        total_batches += 1
        metrics = _compute_classification_metrics(all_targets, all_preds)
        progress_bar.set_postfix(
            loss=f"{running_loss / total_batches:.4f}",
            acc=f"{metrics['acc']:.4f}",
            f1=f"{metrics['f1']:.4f}",
        )

    metrics = _compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(total_batches, 1)
    return metrics


@torch.no_grad()
def predict_classes(model, dataloader, device, stage="Predict", return_probabilities=False):
    model.eval()
    targets = []
    predictions = []
    probabilities = []

    progress_bar = tqdm(dataloader, desc=stage, leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = torch.argmax(probs, dim=1).tolist()

        predictions.extend(preds)
        probabilities.extend(probs.tolist())
        targets.extend(labels.long().cpu().tolist())

    if return_probabilities:
        return targets, predictions, probabilities
    return targets, predictions


def predict_binary(*args, **kwargs):
    return predict_classes(*args, **kwargs)


def build_confusion_matrix(targets, predictions, labels=None):
    if labels is None:
        labels = sorted(set(targets) | set(predictions))
    return confusion_matrix(targets, predictions, labels=labels).tolist()


def compute_per_class_metrics(targets, predictions, labels=None, class_names=None):
    if labels is None:
        labels = sorted(set(targets) | set(predictions))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for idx, label in enumerate(labels):
        class_name = class_names[label] if class_names and isinstance(label, int) and label < len(class_names) else label
        rows.append(
            {
                "class": label,
                "class_name": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows


def fit_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    patience=5,
    freeze_backbone_epochs=0,
    class_labels=None,
    class_names=None,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    if freeze_backbone_epochs > 0:
        trainable_params = freeze_backbone_keep_head_trainable(model)
        print(
            f"Backbone congelado nas primeiras {freeze_backbone_epochs} epocas | "
            f"parametros treinaveis={trainable_params}"
        )

    for epoch in range(num_epochs):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            unfreeze_all_layers(model)
            print(f"Todas as camadas liberadas a partir da epoca {epoch + 1}.")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            stage="Val",
            epoch=epoch,
            num_epochs=num_epochs,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_metrics["f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"train_precision={train_metrics['precision']:.4f} | "
            f"train_recall={train_metrics['recall']:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"lr={current_lr:.8f}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping na epoca {epoch + 1} | "
                f"melhor val_f1={best_val_f1:.4f} na epoca {best_epoch + 1}"
            )
            break

    model.load_state_dict(best_state)

    print("\nGerando classification report do melhor modelo no conjunto de validacao...")
    val_targets, val_preds = predict_classes(model, val_loader, device, stage="Evaluation")
    report = classification_report(
        val_targets,
        val_preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0,
    )
    print(report)

    best_metrics = {
        "epoch": best_epoch,
        "val_loss": history["val_loss"][best_epoch],
        "val_acc": history["val_acc"][best_epoch],
        "val_precision": history["val_precision"][best_epoch],
        "val_recall": history["val_recall"][best_epoch],
        "val_f1": history["val_f1"][best_epoch],
    }

    return model, history, best_metrics
