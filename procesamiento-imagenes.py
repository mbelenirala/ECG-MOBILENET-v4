import os
import cv2
import numpy as np
import pandas as pd
import ast
from glob import glob
from tqdm import tqdm

# === CONFIGURACIÓN DE RUTAS ===
csv_path = 'C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/ptbxl_database_singleclass.csv'
images_root = 'C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/archive'

# Nuevo dataset 3 clases
output_folder = 'C:/Users/belen/OneDrive/Escritorio/MOBILENET-V4/dataset_3clases'
output_dirs = {
    "NORMAL": os.path.join(output_folder, "NORMAL"),
    "ANORMAL": os.path.join(output_folder, "ANORMAL"),
    "MI": os.path.join(output_folder, "MI")
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# === CARGAR CSV ===
df = pd.read_csv(csv_path)
df['superdiagnostic_class'] = df['superdiagnostic_class'].apply(ast.literal_eval)

# === INDEXAR TODAS LAS IMÁGENES DISPONIBLES ===
all_images = glob(os.path.join(images_root, '**', '*_lr-0.png'), recursive=True)
image_index = {os.path.basename(path): path for path in all_images}

copiados = 0
faltantes = []

# === FUNCIÓN: EXTRAER SOLO TRAZO OSCURO ===
def extraer_trazo_oscuro(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([180, 80, 110], dtype=np.uint8)  # rango tolerante
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_cleaned

# === FUNCIÓN: RECONSTRUIR ECG EN FONDO BLANCO ===
def reconstruir_ecg_con_fondo_blanco(trazo_binario):
    fondo = np.ones((*trazo_binario.shape, 3), dtype=np.uint8) * 255
    fondo[trazo_binario > 0] = (0, 0, 0)
    return fondo

# === PROCESAMIENTO DE IMÁGENES ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    record_id = row['ecg_id']
    clase_original = row['superdiagnostic_class'][0] if row['superdiagnostic_class'] else None
    image_filename = f"{record_id}_lr-0.png"

    if image_filename not in image_index or not clase_original:
        faltantes.append(image_filename)
        continue

    # Asignación a 3 clases
    if clase_original == "NORM":
        clase_final = "NORMAL"
    elif clase_original == "MI":
        clase_final = "MI"
    else:
        clase_final = "ANORMAL"

    input_path = image_index[image_filename]
    image = cv2.imread(input_path)
    if image is None:
        faltantes.append(image_filename)
        continue

    try:
        # --- Corrección de perspectiva ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, edged = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        warped = image.copy()

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect = np.array([
                    pts[np.argmin(s)],
                    pts[np.argmin(diff)],
                    pts[np.argmax(s)],
                    pts[np.argmax(diff)]
                ], dtype="float32")
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxWidth = max(int(widthA), int(widthB))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                break

        # === EXTRAER TRAZO ===
        final_clean = extraer_trazo_oscuro(warped)

        # === RECORTE Y SEGMENTACIÓN 3x4 ===
        final_clean = final_clean[219:, :]
        rows, cols = 3, 4
        h, w = final_clean.shape
        lead_h = h // rows
        lead_w = w // cols
        leads = []
        for r in range(rows):
            for c in range(cols):
                lead = final_clean[r*lead_h:(r+1)*lead_h, c*lead_w:(c+1)*lead_w]
                lead_resized = cv2.resize(lead, (224, 224))
                leads.append(lead_resized)
        row_imgs = [np.hstack(leads[i*cols:(i+1)*cols]) for i in range(rows)]
        final_ecg_layout = np.vstack(row_imgs)

        # === CONSTRUIR IMAGEN RGB CON FONDO BLANCO ===
        ecg_rgb = reconstruir_ecg_con_fondo_blanco(final_ecg_layout)

        # === GUARDAR ===
        output_filename = f"{record_id}_clean.png"
        output_path = os.path.join(output_dirs[clase_final], output_filename)
        cv2.imwrite(output_path, ecg_rgb)
        copiados += 1

    except Exception as e:
        print(f"⚠️ Error procesando {image_filename}: {e}")
        faltantes.append(image_filename)

# === RESUMEN ===
print(f"\n✅ Total imágenes procesadas y guardadas: {copiados}")
if faltantes:
    print(f"⚠️ Imágenes faltantes o con error: {len(faltantes)}")
    with open('faltantes.txt', 'w') as f:
        for fname in faltantes:
            f.write(fname + '\n')
