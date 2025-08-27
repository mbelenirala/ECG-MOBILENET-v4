#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento MobileNet-V4 para clasificación de ECG (3 clases: NORMAL, ANORMAL, MI)
- PyTorch + timm
- Focal Loss + class_weight (derivados del set de train)
- AMP seguro (habilita en CUDA, usa bfloat16 en CPU)
- ReduceLROnPlateau + EarlyStopping
- Guarda mejor modelo, historia, reporte y matriz de confusión
Requisitos:
    pip install torch torchvision timm scikit-learn opencv-python matplotlib
Uso (PowerShell, una línea):
    python C:/Users/belen/mobilenet-v4/train_mobilenetv4_ecg.py --data_dir "C:/Users/belen/MOBILENET-V4/dataset_3clases" --out_dir "C:/Users/belen/MOBILENET-V4/runs_mnv4" --epochs 25 --batch_size 32 --lr 1e-4 --image_size 224 --model_name mobilenetv4_hybrid_medium
"""
import os
from datetime import datetime
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
from sklearn.model_selection import StratifiedShuffleSplit

import timm
import matplotlib.pyplot as plt


# -----------------------------
# Config & utilidades
# -----------------------------
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "./runs_mnv4"
    model_name: str = "mobilenetv4_hybrid_medium"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 4
    patience: int = 8  # early stopping
    focal_alpha: float = 0.25  # no se usa directamente (usamos class_weight)
    focal_gamma: float = 3.0   # foco en ejemplos difíciles
    freeze_backbone_epochs: int = 3  # calentar la cabeza antes de FT total
    balance_from_train: bool = True  # usar class_weight del set de train


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
        self.alpha = alpha  # tensor de pesos por clase o float
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
# Split estratificado (train/val/test)
# -----------------------------
def stratified_splits(imagefolder_dataset, val_split=0.15, test_split=0.15, seed=42):
    targets = [y for _, y in imagefolder_dataset.samples]
    idxs = np.arange(len(targets))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idxs, targets))

    targets_trainval = [targets[i] for i in trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split/(1.0 - test_split), random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_idx, targets_trainval))

    train_idx = [trainval_idx[i] for i in train_idx]
    val_idx = [trainval_idx[i] for i in val_idx]

    return train_idx, val_idx, test_idx


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), shear=1),
        transforms.ToTensor(),
        # Normalización estándar de ImageNet; aunque sean trazos en B/N, ayuda al TL
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
# Entrenamiento / Validación
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, amp_dtype, amp_enabled):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_dtype, amp_enabled):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets, all_preds = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
    return total_loss / total, correct / total, np.array(all_targets), np.array(all_preds)


# -----------------------------
# Guardar matriz de confusión
# -----------------------------
def save_confusion_matrix(cm, class_names, out_png, title="Matriz de confusión"):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest')
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Carpeta con las subcarpetas NORMAL/ANORMAL/MI")
    parser.add_argument("--out_dir", type=str, default="./runs_mnv4")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="mobilenetv4_hybrid_medium")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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
        seed=args.seed
    )
    # Crear subcarpeta única con timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.out_dir = os.path.join(cfg.out_dir, timestamp)
    ensure_dir(cfg.out_dir)
    print(f">> Resultados se guardarán en: {cfg.out_dir}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    amp_dtype   = torch.float16 if amp_enabled else torch.bfloat16
    ensure_dir(cfg.out_dir)

    # Datasets
    train_tfms, val_tfms = build_transforms(cfg.image_size)
    full_ds = datasets.ImageFolder(cfg.data_dir, transform=val_tfms)  # se re-asigna transform por split

    # Splits
    train_idx, val_idx, test_idx = stratified_splits(full_ds, val_split=cfg.val_split, test_split=cfg.test_split, seed=cfg.seed)
    train_ds = Subset(datasets.ImageFolder(cfg.data_dir, transform=train_tfms), train_idx)
    val_ds   = Subset(datasets.ImageFolder(cfg.data_dir, transform=val_tfms), val_idx)
    test_ds  = Subset(datasets.ImageFolder(cfg.data_dir, transform=val_tfms), test_idx)

    class_names = datasets.ImageFolder(cfg.data_dir).classes
    print("Clases:", class_names)

    # Pesos por clase (desde train)
    if cfg.balance_from_train:
        labels_train = [datasets.ImageFolder(cfg.data_dir).samples[i][1] for i in train_idx]
        class_counts = np.bincount(labels_train, minlength=len(class_names))
        print("Conteo por clase (train):", class_counts.tolist())
        class_weights = class_counts.sum() / (len(class_counts) * class_counts + 1e-6)
        alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # --- AUMENTO ESPECÍFICO PARA MI ---
        if "MI" in class_names:
            mi_idx = class_names.index("MI")
            boost = 2.5  # probá 2.0–3.0
            alpha_tensor[mi_idx] = alpha_tensor[mi_idx] * boost
            print(f"Pesos alpha (con boost a MI x{boost}):", alpha_tensor.tolist())
    else:
        alpha_tensor = None


    # Dataloaders
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    # Modelo
    num_classes = len(class_names)
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Optim / Loss / LR scheduler
    loss_fn = FocalLoss(alpha=alpha_tensor, gamma=cfg.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

    # Congelar backbone por algunas épocas (warmup de la cabeza)
    def set_backbone_requires_grad(flag: bool):
        # Asumimos que el clasificador es model.get_classifier() en timm
        for n, p in model.named_parameters():
            if "classifier" in n or "head" in n or "fc" in n:
                p.requires_grad = True
            else:
                p.requires_grad = flag

    set_backbone_requires_grad(False)  # primero solo la cabeza
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    early = EarlyStopping(patience=cfg.patience, minimize=True)

    best_val_loss = float('inf')
    best_path = os.path.join(cfg.out_dir, f"{cfg.model_name}_best.pt")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(">> Iniciando script...")
    print(f">> data_dir = {cfg.data_dir}")
    print(">> Construyendo transforms y dataset (ImageFolder)...")
    full_ds = datasets.ImageFolder(cfg.data_dir, transform=val_tfms)
    print(f">> Dataset listo. Total imágenes: {len(full_ds)}. Clases: {full_ds.classes}")
    print(">> Haciendo splits estratificados (train/val/test)...")
    
    for epoch in range(cfg.epochs):
        t0 = time.time()

        # Unfreeze total a partir de cierto punto
        if epoch == cfg.freeze_backbone_epochs:
            set_backbone_requires_grad(True)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, amp_dtype, amp_enabled)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device, amp_dtype, amp_enabled)
        scheduler.step(val_loss)

        # Guardar mejor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_name": cfg.model_name,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "cfg": asdict(cfg)
            }, best_path)

        # Early stopping
        early.step(val_loss)
        elapsed = time.time() - t0

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"lr={current_lr:.2e} | {elapsed:.1f}s")

        if early.should_stop:
            print(f"⏹️ Early stopping activado en epoch {epoch+1}. Mejor val_loss: {best_val_loss:.4f}")
            break

    # Guardar historia
    hist_path = os.path.join(cfg.out_dir, f"{cfg.model_name}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Evaluación final en TEST con el mejor checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, loss_fn, device, amp_dtype, amp_enabled)
    print(f"\n[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Reporte y matriz
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nReporte de clasificación (TEST):\n", report)
    report_path = os.path.join(cfg.out_dir, f"{cfg.model_name}_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(cfg.out_dir, f"{cfg.model_name}_cm.png")
    save_confusion_matrix(cm, class_names, cm_path, title="Confusión (TEST)")

    # Guardar también script de inferencia rápida
    infer_path = os.path.join(cfg.out_dir, "inferencia_rapida.py")
    with open(infer_path, "w", encoding="utf-8") as f:
        f.write(f"""# -*- coding: utf-8 -*-
import torch
from PIL import Image
from torchvision import transforms
import timm

ckpt_path = r\"{best_path}\"
ckpt = torch.load(ckpt_path, map_location='cpu')
model_name = ckpt['model_name']
class_names = ckpt['class_names']

model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
model.load_state_dict(ckpt['state_dict'])
model.eval()

tfm = transforms.Compose([
    transforms.Resize({cfg.image_size + int(cfg.image_size*0.10)}),
    transforms.CenterCrop({cfg.image_size}),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predecir(path_img: str):
    img = Image.open(path_img).convert('RGB')
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = prob.argmax().item()
    return class_names[idx], prob[idx].item()

if __name__ == "__main__":
    # print(predecir("C:/ruta/a/una/imagen.png"))
    pass
""")

    print("\nArchivos guardados en:", os.path.abspath(cfg.out_dir))
    print(" - Mejor modelo:", best_path)
    print(" - Historia:", hist_path)
    print(" - Reporte TEST:", report_path)
    print(" - Matriz de confusión:", cm_path)
    print(" - Script inferencia:", infer_path)


if __name__ == "__main__":
    main()
