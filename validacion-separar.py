import os
import random
import shutil

# === CONFIG ===
base_dir = r"C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/dataset_3clases"
val_dir = r"C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/dataset_validacion_manual"
classes = ["NORMAL", "ANORMAL", "MI"]
n_move = 150

os.makedirs(val_dir, exist_ok=True)

for cls in classes:
    src_dir = os.path.join(base_dir, cls)
    dst_dir = os.path.join(val_dir, cls)
    os.makedirs(dst_dir, exist_ok=True)

    imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(imgs) < n_move:
        print(f"⚠️ Clase {cls} tiene solo {len(imgs)} imágenes, se moverán todas.")
        chosen = imgs
    else:
        chosen = random.sample(imgs, n_move)

    for fname in chosen:
        shutil.move(os.path.join(src_dir, fname),
                    os.path.join(dst_dir, fname))

    print(f"✅ Movidas {len(chosen)} imágenes de {cls} a {dst_dir}")

print("\nFinalizado. Imágenes de validación manual guardadas en:", val_dir)
