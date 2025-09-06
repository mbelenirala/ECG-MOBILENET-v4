import os, json, torch, timm
import numpy as np
from PIL import Image
from torchvision import transforms

class StableEnsemble:
    def __init__(self, bundle_dir, device=None):
        self.bundle_dir = bundle_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar config y thresholds
        with open(os.path.join(bundle_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(bundle_dir, "thresholds.json")) as f:
            thcfg = json.load(f)
        with open(os.path.join(bundle_dir, "models.json")) as f:
            mcfg = json.load(f)

        self.class_names = cfg["class_names"]
        self.C = len(self.class_names)
        self.thresholds = np.array([thcfg["thresholds"][c] for c in self.class_names], dtype=np.float32)

        # Transforms coherentes con el entrenamiento
        tfms = [
            transforms.Resize(int(cfg["image_size"] * 1.10)),
            transforms.CenterCrop(cfg["image_size"]),
        ]
        if cfg.get("grayscale", True):
            tfms.append(transforms.Grayscale(num_output_channels=3))
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalize_mean"], std=cfg["normalize_std"]),
        ]
        self.tfm = transforms.Compose(tfms)

        # Cargar modelos
        self.models = []
        for fname in mcfg["checkpoints"]:
            ck = torch.load(os.path.join(bundle_dir, fname), map_location=self.device)
            m = timm.create_model(ck["model_name"], pretrained=False, num_classes=len(ck["class_names"]))
            m.load_state_dict(ck["state_dict"])
            m.eval().to(self.device)
            self.models.append(m)
        if not self.models:
            raise RuntimeError("No hay checkpoints en el bundle.")

    @torch.no_grad()
    def predict_image(self, path_img):
        img = Image.open(path_img).convert("RGB")
        x = self.tfm(img).unsqueeze(0).to(self.device)
        probs = []
        for m in self.models:
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device=="cuda")):
                p = torch.softmax(m(x), dim=1).cpu().numpy()[0]
            probs.append(p)
        p = np.mean(np.stack(probs,0), axis=0)  # promedio de modelos

        # Reglas de decisión por umbrales (one-vs-rest)
        candidates = p >= self.thresholds
        if candidates.any():
            idx = int(np.argmax(np.where(candidates, p, -1.0)))
        else:
            idx = int(np.argmax(p))
        return self.class_names[idx], float(p[idx]), p

# CLI opcional:
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    se = StableEnsemble(args.bundle_dir)
    cls, conf, p = se.predict_image(args.image)
    print(cls, conf)


#esto para probar con una imagen
#from stable_infer import StableEnsemble
#se = StableEnsemble("/content/drive/MyDrive/modelo_estable_ensemble")
# se.predict_image("/content/ECG-MOBILENET-v4/ejemplo.png")
