import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# ===========================================
# GradCAM Class
# ===========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Mengambil gradient output
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        # ERROR FIX 1: "element 0 does not require grad"
        # Kita buat clone dari tensor input, lepaskan dari graph sebelumnya (detach),
        # lalu paksa dia untuk mencatat gradient (requires_grad=True).
        # Ini agar GradCAM tetap jalan walau caller function pakai torch.no_grad()
        input_model = input_tensor.detach().clone()
        input_model.requires_grad = True

        # Paksa PyTorch mencatat operasi gradient di blok ini
        with torch.enable_grad():
            output = self.model(input_model)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            # Backward Pass
            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward()

            # Ambil gradients & activations dari hooks
            gradients = self.gradients
            activations = self.activations
            
            # Global Average Pooling pada Gradients [Batch, Channel, H, W] -> [Batch, Channel, 1, 1]
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Weighted combination
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize min-max
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            
            # ERROR FIX 2: "Can't call numpy() on Tensor that requires grad"
            # Kita harus detach() dulu sebelum convert ke numpy
            return cam.detach().cpu().numpy()[0, 0]

# ===========================================
# Helper: Tensor to Image (Denormalization)
# ===========================================
def tensor_to_image(tensor):
    """Mengubah tensor (normalized) kembali ke gambar numpy uint8"""
    # Fix Error Numpy: Selalu detach dan pindah ke cpu sebelum convert
    tensor = tensor.clone().detach().cpu()
    
    if tensor.dim() == 4:
        tensor = tensor[0]
        
    # Denormalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to Numpy (H, W, C)
    img_np = tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

# ===========================================
# Helper: Overlay Heatmap
# ===========================================
def overlay_heatmap_on_image(cam, original_image_np):
    h, w = original_image_np.shape[:2]
    
    # Resize CAM ke ukuran gambar asli
    cam = cv2.resize(cam, (w, h))
    
    # Apply Colormap Jet
    # Konversi ke uint8 0-255
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose (0.4 heatmap, 0.6 original)
    overlay = (0.4 * heatmap) + (0.6 * original_image_np)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

# ===========================================
# MAIN FUNCTION
# ===========================================
def generate_heatmap(input_tensor, model, target_layer):
    """
    Fungsi wrapper utama agar kompatibel dengan pemanggilan di skin_detection.py
    """
    try:
        # 1. Inisialisasi GradCAM
        grad_cam = GradCAM(model, target_layer)
        
        # 2. Generate Raw CAM (Mask)
        # Note: input_tensor tidak perlu diubah device-nya di sini, biarkan apa adanya
        cam_mask = grad_cam(input_tensor)
        
        # 3. Dapatkan Gambar Asli dari Tensor untuk Overlay
        original_img_np = tensor_to_image(input_tensor)
        
        # 4. Buat Overlay Final
        final_pil = overlay_heatmap_on_image(cam_mask, original_img_np)
        
        return final_pil
        
    except Exception as e:
        print(f"[GradCAM Critical Error] {e}")
        # Fallback: Kembalikan gambar asli jika GradCAM gagal total
        try:
            return Image.fromarray(tensor_to_image(input_tensor))
        except:
            return None