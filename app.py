import io
import os
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from utils.orthanc_client import upload_dicom, get_orthanc_preview
from modules.bone_detection import BoneDetector
from modules.brain_detection import BrainTumorDetector
from modules.skin_detection import SkinDetector
from modules.ecg_detection import ECGDetector

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

print("--- Initializing AI Modules ---")
bone_engine = BoneDetector()
brain_engine = BrainTumorDetector()
skin_engine = SkinDetector()
ecg_engine = ECGDetector()
print("--- Initialization Complete ---")

def img_to_b64(img_obj):
    try:
        if img_obj is None: return None
        if img_obj.mode != 'RGB': img_obj = img_obj.convert('RGB')
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image Convert Error: {e}")
        return ""

# FUNGSI TAMBAHAN: Enhancement Citra (Pengganti RealESRGAN jika model belum load)
def enhance_image_cv(pil_img):
    try:
        img_np = np.array(pil_img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # untuk meningkatkan detail tekstur kulit/tulang
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(final_rgb)
    except Exception as e:
        print(f"Enhancement Error: {e}")
        return pil_img

SCABIES_INFO = {
    "description": "Scabies (kudis) adalah penyakit kulit menular yang disebabkan oleh tungau Sarcoptes scabiei.",
    "treatment": "1. Krim Permethrin 5%.<br>2. Cuci pakaian dengan air panas.<br>3. Hindari kontak langsung."
}

@app.route('/')
def index(): return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename): return send_from_directory('static', filename)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        analysis_type = request.form.get("type", "bone") 
        batch_id = request.form.get("batch_id", None)
        file_bytes = file.read()
        filename = file.filename
        
        # --- LOGIKA ECG ---
        if analysis_type == 'ecg':
            options = {
                'rows': request.form.get('ecg_rows', 1),
                'grid': request.form.get('ecg_grid', 'true'),
                'details': request.form.get('ecg_details', 'false')
            }
            label, conf, explanation, plot_image = ecg_engine.predict_from_file(file_bytes, filename, options)
            return jsonify({
                "type": "ecg", "filename": filename, "label": label, "confidence": conf,
                "explanation": explanation, "original_image": plot_image
            })

        # --- LOGIKA GAMBAR ---
        is_dicom = filename.lower().endswith('.dcm')
        image_pil = None
        viewer_url = None
        study_uid = None
        
        if is_dicom:
            orthanc_res = upload_dicom(file_bytes, batch_id)
            if orthanc_res:
                instance_id, study_uid, frames = orthanc_res
                image_pil = get_orthanc_preview(instance_id)
                viewer_url = f"http://localhost:8042/ohif/viewer?StudyInstanceUIDs={study_uid}"
            else:
                return jsonify({'error': 'Orthanc Upload Failed'}), 500
        else:
            try:
                image_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            except Exception as img_err:
                return jsonify({'error': f"File bukan gambar valid: {str(img_err)}"}), 400
        
        result = {
            "type": analysis_type,
            "filename": filename,
            "original_image": img_to_b64(image_pil),
            "viewer_url": viewer_url,
            "study_uid": study_uid
        }

        if analysis_type == 'brain':
            label, conf, annotated, mask, expl, size = brain_engine.predict(image_pil)
            result.update({ "label": label, "confidence": conf, "explanation": expl, "tumor_size": f"{size:.2f}%", "annotated_image": img_to_b64(annotated), "mask_image": img_to_b64(mask) })
            
        elif analysis_type == 'bone':
            # Bone engine sudah mengembalikan enhanced image, tapi kita pastikan ada
            label, conf, heatmap, all_preds, enhanced = bone_engine.predict(image_pil)
            if enhanced is None: enhanced = enhance_image_cv(image_pil) # Fallback
            result.update({ "label": label, "confidence": conf, "all_predictions": all_preds, "gradcam_image": img_to_b64(heatmap), "enhanced_image": img_to_b64(enhanced) })

        elif analysis_type == 'skin':
            label, conf, heatmap = skin_engine.predict(image_pil)
            
            # [FIX] Generate Enhanced Image secara manual di sini untuk fitur Scabies
            enhanced_pil = enhance_image_cv(image_pil)
            
            skin_result = { 
                "label": label, 
                "confidence": conf, 
                "gradcam_image": img_to_b64(heatmap),
                "enhanced_image": img_to_b64(enhanced_pil) # Kirim data enhanced
            }
            if "scabies" in label.lower(): skin_result.update(SCABIES_INFO)
            result.update(skin_result)

        return jsonify(result)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)