import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from utils.gradcam import generate_heatmap

# Pastikan Device konsisten
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinDetector:
    def __init__(self, model_path="Models/skin_model.pth"):
        self.model = None
        self.load_model(model_path)

    def load_model(self, path):
        try:
            if not os.path.exists(path):
                print(f"[Skin] Warning: Weights not found at {path}")
                return

            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            
            # Load checkpoint
            ckpt = torch.load(path, map_location=DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            
            # PINDAHKAN MODEL KE DEVICE (Penting untuk error Input type mismatch)
            self.model.to(DEVICE)
            self.model.eval()
            print(f"[Skin] Model loaded on {DEVICE}.")
        except Exception as e:
            print(f"[Skin] Error: {e}")

    def predict(self, img_pil):
        if not self.model:
            return "Model Error", 0.0, None

        # 1. Siapkan Tensor di Device yang benar
        img_tensor = data_transforms(img_pil).unsqueeze(0).to(DEVICE)
        
        # 2. PENTING UNTUK GRADCAM: Nyalakan requires_grad
        # Error "element 0 ... does not require grad" terjadi karena baris ini kurang
        img_tensor.requires_grad = True
        
        # 3. HAPUS 'with torch.no_grad():' 
        # Kita butuh gradient untuk GradCAM, jadi biarkan gradient mengalir
        try:
            # Pastikan model di mode eval tapi parameter tetap ada
            self.model.zero_grad()
            
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf_score, preds = torch.max(probs, 1)
            
            label = "Scabies" if preds.item() == 1 else "Healthy Skin"
            confidence = conf_score.item() * 100
            
            # 4. Generate Heatmap
            # Pastikan generate_heatmap menangani backward pass
            heatmap_pil = generate_heatmap(img_tensor, self.model, self.model.layer4[-1])
            
            return label, confidence, heatmap_pil

        except Exception as e:
            print(f"Prediction Error: {e}")
            # Fallback jika error, kembalikan label tanpa heatmap
            return "Error", 0.0, None