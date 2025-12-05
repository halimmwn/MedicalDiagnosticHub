import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from utils.gradcam import generate_heatmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BoneDetector:
    def __init__(self, model_path="Models/bone_best.pth"):
        self.model = None
        self.classes = ['Healthy', 'Fracture', 'Avulsion', 'Comminuted', 'Greenstick', 'Hairline', 'Impacted', 'Longitudinal', 'Oblique', 'Spiral']
        self.load_model(model_path)

    def load_model(self, path):
        try:
            if not os.path.exists(path):
                print(f"[Bone] Warning: Weights not found at {path}. Using empty model.")
                return

            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(self.model.fc.in_features, 512),
                nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, len(self.classes))
            )
            
            # Load checkpoint
            ckpt = torch.load(path, map_location=DEVICE)
            
            if 'classes' in ckpt: self.classes = ckpt['classes']
            
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            
            # PENTING: Pindahkan ke DEVICE dan set eval
            self.model.to(DEVICE)
            self.model.eval()
            print("[Bone] Model loaded successfully.")
        except Exception as e:
            print(f"[Bone] Error loading model: {e}")
            self.model = None

    def enhance_image(self, img_pil):
        """
        Applies enhancement (CLAHE/OpenCV) to simulate GAN-like clarity 
        """
        try:
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced_cv = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            return Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Enhancement Error: {e}")
            return img_pil

    def predict(self, img_pil):
        if not self.model:
            return "Model Error", 0.0, None, {}, img_pil

        # 1. Enhance Image
        enhanced_pil = self.enhance_image(img_pil)

        # 2. Prepare Tensor & Move to Device
        img_tensor = data_transforms(img_pil).unsqueeze(0).to(DEVICE)
        
        # PERBAIKAN 1: Wajib set True agar GradCAM bisa jalan
        img_tensor.requires_grad = True
        
        # 3. Predict
        try:
            # PERBAIKAN 2: HAPUS 'with torch.no_grad():'
            # Kita biarkan gradient mengalir untuk GradCAM
            
            self.model.zero_grad() # Reset gradient sebelumnya
            
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf_score, preds = torch.max(probs, 1)
            
            label = self.classes[preds.item()]
            confidence = conf_score.item() * 100
            
            all_predictions = {c: f"{probs[0][i].item()*100:.1f}" for i, c in enumerate(self.classes)}
            
            # 4. Generate Heatmap (Sekarang akan berhasil karena ada gradient)
            heatmap_pil = generate_heatmap(img_tensor, self.model, self.model.layer4[-1])
            
            return label, confidence, heatmap_pil, all_predictions, enhanced_pil
            
        except Exception as e:
            print(f"[Bone Prediction Error] {e}")
            # Return safe values on error
            return "Error", 0.0, None, {}, enhanced_pil