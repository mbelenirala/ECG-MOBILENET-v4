import os, torch, timm
import numpy as np
from PIL import Image
from torchvision import transforms

class EnsembleECG:
    def __init__(self, data_dir, out_dir_k, image_size=224, device=None):
        self.data_dir = data_dir
        self.out_dir_k = out_dir_k
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # classes desde un ImageFolder
        from torchvision import datasets
        self.class_names = datasets.ImageFolder(self.data_dir).classes
        self.C = len(self.class_names)
        self.mi_idx = self.class_names.index("MI")
        # tfms (coherentes con el entrenamiento)
        self.tfm = transforms.Compose([
            transforms.Resize(int(self.image_size*1.10)),
            transforms.CenterCrop(self.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        # cargar ckpts
        self.models = []
        for fd in sorted(os.listdir(self.out_dir_k)):
            if not fd.startswith("fold_"): continue
            d = os.path.join(self.out_dir_k, fd)
            for f in os.listdir(d):
                if f.endswith("_best.pt"):
                    ck = os.path.join(d,f)
                    ckpt = torch.load(ck, map_location=self.device)
                    m = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["class_names"]))
                    m.load_state_dict(ckpt["state_dict"])
                    m.eval().to(self.device)
                    self.models.append(m)
        if not self.models:
            raise RuntimeError("No encontré checkpoints *_best.pt en " + self.out_dir_k)

    @torch.no_grad()
    def predict_image(self, path_img, mi_threshold=None):
        img = Image.open(path_img).convert("RGB")
        x = self.tfm(img).unsqueeze(0).to(self.device)
        # promedio de probabilidades (softmax) entre modelos
        probs = []
        for m in self.models:
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device=="cuda")):
                p = torch.softmax(m(x), dim=1).cpu().numpy()[0]
            probs.append(p)
        p = np.mean(np.stack(probs,0), axis=0)
        # umbral opcional para MI
        if mi_threshold is not None and p[self.mi_idx] >= mi_threshold:
            idx = self.mi_idx
        else:
            idx = int(p.argmax())
        return self.class_names[idx], float(p[idx]), p

# uso rápido:
# ens = EnsembleECG("/content/ECG-MOBILENET-v4/dataset_3clases", "/content/drive/MyDrive/runs_kfold_balanced")
# print(ens.predict_image("/ruta/a/una_imagen.png"))
