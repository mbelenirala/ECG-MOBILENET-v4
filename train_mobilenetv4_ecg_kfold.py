#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K-Fold Cross-Validation con MobileNet-V4 para clasificación de ECG (3 clases: NORMAL, ANORMAL, MI)
- PyTorch + timm
- Focal Loss + class_weight por fold
- AMP (mixed precision), ReduceLROnPlateau, EarlyStopping
- Split fijo de TEST y k-fold en TRAIN+VAL
- Guarda por fold: mejor modelo, reporte y matriz de confusión (VAL y TEST)
- Al final: resumen (media y desvío) y opción de entrenar modelo final en TRAIN+VAL y evaluar en TEST

Requisitos:
    pip install torch torchvision timm scikit-learn opencv-python matplotlib
Uso ejemplo (Windows):
    python train_mobilenetv4_ecg_kfold.py ^
      --data_dir "C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/dataset_3clases" ^
      --out_dir  "C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/runs_kfold" ^
      --k_folds 5 --epochs 20 --batch_size 32 --lr 1e-4 --image_size 224
"""
import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import timm
import matplotlib.pyplot as plt


# -----------------------------
# Config & utilidades
# -----------------------------
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "./runs_kfold"
    model_name: str = "mobilenetv4_hybrid_medium"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    k_folds: int = 5
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 4
    patience: int = 8  # early stopping
    focal_gamma: float = 2.0
    freeze_backbone_epochs: int = 3  # calentar la cabeza antes de FT total
    final_fit: bool = True  # Entrenar modelo final en TRAIN+VAL y evaluar en TEST


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Focal Loss con class_weight
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # tensor de pesos por clase
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), shear=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.10)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


# -----------------------------
# EarlyStopping simple
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=8, minimize=True):
        self.patience = patience
        self.best_score = math.inf if minimize else -math.inf
        self.counter = 0
        self.should_stop = False
        self.minimize = minimize

    def step(self, value):
        improved = (value < self.best_score) if self.minimize else (value > self.best_score)
        if improved:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# -----------------------------
# Loop de entrenamiento/validación
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
    return total_loss / total, correct / total, np.array(all_targets), np.array(all_preds)


def save_confusion_matrix(cm, class_names, out_png, title="Matriz de confusión"):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")
    fig.tight_layout()
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')
    plt.savefig(out_png, bbox_inches='tight', dpi=160)
    plt.close(fig)


def compute_class_weights_from_indices(all_samples, indices, num_classes):
    labels = [all_samples[i][1] for i in indices]
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (len(counts) * counts + 1e-6)
    return weights, counts.tolist()


def make_model(model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Carpeta con NORMAL/ANORMAL/MI")
    parser.add_argument("--out_dir", type=str, default="./runs_kfold")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="mobilenetv4_hybrid_medium")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_fit", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        model_name=args.model_name,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        k_folds=args.k_folds,
        test_split=args.test_split,
        seed=args.seed,
        final_fit=args.final_fit or True
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.out_dir)

    # Datasets base y transforms
    train_tfms, val_tfms = build_transforms(cfg.image_size)
    full_ds_val = datasets.ImageFolder(cfg.data_dir, transform=val_tfms)    # para splits / val/test
    class_names = full_ds_val.classes
    num_classes = len(class_names)
    print("Clases:", class_names)

    # Split fijo de TEST (estratificado)
    targets = [y for _, y in full_ds_val.samples]
    idxs = np.arange(len(targets))
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_split, random_state=cfg.seed)
    trainval_idx, test_idx = next(sss_test.split(idxs, targets))
    print(f"Total: {len(idxs)} | Train+Val: {len(trainval_idx)} | Test: {len(test_idx)}")

    # K-Fold en TRAIN+VAL
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    y_trainval = [targets[i] for i in trainval_idx]

    fold_summaries = []
    # Prepara DataLoader de TEST (constante)
    test_ds = Subset(datasets.ImageFolder(cfg.data_dir, transform=val_tfms), test_idx)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    for fold, (tr_rel, val_rel) in enumerate(skf.split(trainval_idx, y_trainval), start=1):
        print(f"\n===== Fold {fold}/{cfg.k_folds} =====")
        tr_idx = [trainval_idx[i] for i in tr_rel]
        va_idx = [trainval_idx[i] for i in val_rel]

        # Datasets por fold
        ds_train = Subset(datasets.ImageFolder(cfg.data_dir, transform=train_tfms), tr_idx)
        ds_val   = Subset(datasets.ImageFolder(cfg.data_dir, transform=val_tfms), va_idx)

        # Pesos por clase a partir de TRAIN del fold
        class_weights, counts = compute_class_weights_from_indices(datasets.ImageFolder(cfg.data_dir).samples,
                                                                   tr_idx, num_classes)
        alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print("Conteo por clase (train fold):", counts)

        # Dataloaders
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
        dl_val   = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

        # Modelo y optimizadores
        model = make_model(cfg.model_name, num_classes, device)

        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=cfg.focal_gamma)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

        # Congelar backbone (warm-up de la cabeza)
        def set_backbone_requires_grad(flag: bool):
            for n, p in model.named_parameters():
                if "classifier" in n or "head" in n or "fc" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = flag

        set_backbone_requires_grad(False)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        early = EarlyStopping(patience=cfg.patience, minimize=True)

        best_val_loss = float('inf')
        fold_dir = os.path.join(cfg.out_dir, f"fold_{fold:02d}")
        ensure_dir(fold_dir)
        best_path = os.path.join(fold_dir, f"{cfg.model_name}_fold{fold:02d}_best.pt")

        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

        for epoch in range(cfg.epochs):
            t0 = time.time()

            # Unfreeze total a partir de cierto punto
            if epoch == cfg.freeze_backbone_epochs:
                set_backbone_requires_grad(True)

            tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, scaler)
            va_loss, va_acc, y_true_val, y_pred_val = evaluate(model, dl_val, loss_fn, device)
            scheduler.step(va_loss)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save({
                    "model_name": cfg.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "cfg": asdict(cfg)
                }, best_path)

            early.step(va_loss)
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch+1)
            history["train_loss"].append(float(tr_loss))
            history["train_acc"].append(float(tr_acc))
            history["val_loss"].append(float(va_loss))
            history["val_acc"].append(float(va_acc))
            history["lr"].append(float(current_lr))

            print(f"Fold {fold:02d} | Epoch {epoch+1:03d}/{cfg.epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
                  f"lr={current_lr:.2e} | {elapsed:.1f}s")

            if early.should_stop:
                print(f"⏹️ Early stopping en fold {fold} epoch {epoch+1}. Mejor val_loss: {best_val_loss:.4f}")
                break

        # Cargar mejor modelo del fold
        ckpt = torch.load(best_path, map_source='cpu') if not torch.cuda.is_available() else torch.load(best_path)
        model.load_state_dict(ckpt["state_dict"])

        # Evaluación en VAL (mejor checkpoint)
        val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, dl_val, loss_fn, device)
        val_report = classification_report(y_true_val, y_pred_val, target_names=class_names, digits=4, zero_division=0)
        cm_val = confusion_matrix(y_true_val, y_pred_val)
        # Guardar artefactos de VAL
        with open(os.path.join(fold_dir, "val_report.txt"), "w", encoding="utf-8") as f:
            f.write(val_report)
        save_confusion_matrix(cm_val, class_names, os.path.join(fold_dir, "val_cm.png"), title="Confusión (VAL)")

        # Evaluación en TEST (referencia)
        test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, loss_fn, device)
        test_report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4, zero_division=0)
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        with open(os.path.join(fold_dir, "test_report.txt"), "w", encoding="utf-8") as f:
            f.write(test_report)
        save_confusion_matrix(cm_test, class_names, os.path.join(fold_dir, "test_cm.png"), title="Confusión (TEST ref)")

        # Resumen por fold
        fold_summary = {
            "fold": fold,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc)
        }
        # Extra: recall de MI (si existe la clase "MI")
        try:
            # Buscar índice de MI en class_names
            if "MI" in class_names:
                mi_idx = class_names.index("MI")
                # report dict para val
                rep_val = classification_report(y_true_val, y_pred_val, target_names=class_names, output_dict=True, zero_division=0)
                rep_test = classification_report(y_true_test, y_pred_test, target_names=class_names, output_dict=True, zero_division=0)
                fold_summary["val_recall_MI"] = float(rep_val["MI"]["recall"])
                fold_summary["test_recall_MI"] = float(rep_test["MI"]["recall"])
        except Exception:
            pass

        with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(os.path.join(fold_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, indent=2, ensure_ascii=False)

        fold_summaries.append(fold_summary)

    # ---- Resumen global de folds ----
    def _agg(key):
        vals = [fs[key] for fs in fold_summaries if key in fs]
        return {
            "mean": float(np.mean(vals)) if vals else None,
            "std":  float(np.std(vals)) if vals else None
        }

    global_summary = {
        "val_loss": _agg("val_loss"),
        "val_acc": _agg("val_acc"),
        "test_loss": _agg("test_loss"),
        "test_acc": _agg("test_acc"),
        "val_recall_MI": _agg("val_recall_MI"),
        "test_recall_MI": _agg("test_recall_MI"),
        "k_folds": cfg.k_folds
    }
    with open(os.path.join(cfg.out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    # ---- Entrenamiento final (opcional) en TRAIN+VAL completos ----
    if cfg.final_fit:
        print("\n===== Entrenamiento final en TRAIN+VAL completos y evaluación única en TEST =====")
        # Datasets completos
        ds_trainval = Subset(datasets.ImageFolder(cfg.data_dir, transform=train_tfms), trainval_idx)
        dl_trainval = DataLoader(ds_trainval, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, pin_memory=True)

        # Pesos por clase desde TRAIN+VAL completo
        class_weights, counts = compute_class_weights_from_indices(datasets.ImageFolder(cfg.data_dir).samples,
                                                                   trainval_idx, num_classes)
        alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print("Conteo por clase (TRAIN+VAL completo):", counts)

        model = make_model(cfg.model_name, num_classes, device)
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=cfg.focal_gamma)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

        def set_backbone_requires_grad(flag: bool):
            for n, p in model.named_parameters():
                if "classifier" in n or "head" in n or "fc" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = flag

        set_backbone_requires_grad(False)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        early = EarlyStopping(patience=cfg.patience, minimize=True)

        best_path = os.path.join(cfg.out_dir, f"{cfg.model_name}_FINAL_best.pt")
        best_loss = float('inf')

        for epoch in range(cfg.epochs):
            if epoch == cfg.freeze_backbone_epochs:
                set_backbone_requires_grad(True)
            tr_loss, tr_acc = train_one_epoch(model, dl_trainval, optimizer, loss_fn, device, scaler)
            # usar parte de TRAIN+VAL como pseudo-val? aquí usamos TEST solo al final para no sesgar
            scheduler.step(tr_loss)
            print(f"[FINAL] Epoch {epoch+1:03d}/{cfg.epochs} | loss={tr_loss:.4f} acc={tr_acc:.4f}")
            if tr_loss < best_loss:
                best_loss = tr_loss
                torch.save({
                    "model_name": cfg.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "cfg": asdict(cfg)
                }, best_path)
            if early.should_stop:
                break

        # Evaluación única en TEST con el mejor modelo final
        ckpt = torch.load(best_path, map_source='cpu') if not torch.cuda.is_available() else torch.load(best_path)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, loss_fn, device)
        test_report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4, zero_division=0)
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        with open(os.path.join(cfg.out_dir, "FINAL_test_report.txt"), "w", encoding="utf-8") as f:
            f.write(test_report)
        save_confusion_matrix(cm_test, class_names, os.path.join(cfg.out_dir, "FINAL_test_cm.png"),
                              title="Confusión (TEST único)")
        print(f"\n[FINAL on TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    print("\nHecho. Artefactos guardados en:", os.path.abspath(cfg.out_dir))


if __name__ == "__main__":
    main()
