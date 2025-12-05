import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io
import base64
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ARSITEKTUR MODEL 1D
# ============================================================
class ECGNet1D(nn.Module):
    def __init__(self):
        super(ECGNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16) 
        self.fc1 = nn.Linear(32 * 16, 32) 
        self.fc2 = nn.Linear(32, 5) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ECGDetector:
    def __init__(self, model_path="Models/heartbeatfor_model.pt"):
        self.model = None
        self.classes_map = {
            0: "Normal Sinus Rhythm",
            1: "Supraventricular (S) - Indikasi Tachycardia",
            2: "Ventricular Ectopic (V) - Abnormal",
            3: "Fusion Beat (F)",
            4: "Normal Sinus Rhythm with Artefacts"
        }
        self.load_model(model_path)

    def load_model(self, path):
        try:
            if not os.path.exists(path):
                print(f"[ECG] Warning: Weights not found at {path}")
                return
            self.model = ECGNet1D()
            ckpt = torch.load(path, map_location=DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.model.to(DEVICE).eval()
            print("[ECG] Model loaded.")
        except Exception as e:
            print(f"[ECG] Error loading: {e}")
            self.model = None

    def parse_file_to_signal(self, file_bytes, filename=""):
        try:
            content = file_bytes.decode('utf-8')
            data_str = content.replace(',', ' ').split()
            signal = np.array([float(x) for x in data_str if x.replace('.', '', 1).isdigit() or x.lstrip('-').replace('.', '', 1).isdigit()])
            return signal
        except UnicodeDecodeError:
            try:
                signal = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32)
                if len(signal) > 1000:
                    return signal
                return None
            except Exception as bin_err:
                print(f"[Binary Parse Error] {bin_err}")
                return None
        except Exception as e:
            print(f"[ECG Parsing Error] {e}")
            return None

    def extract_heartbeat(self, full_signal, target_len=187):
        """
        LOGIKA CERDAS: Mencari detak jantung berdasarkan kemiringan (Slope), bukan hanya ketinggian.
        Ini menghindari noise/artifact besar yang sering muncul di awal/akhir rekaman.
        """
        try:
            # 1. Potong Margin (Abaikan 5% data awal dan akhir yang sering berisi noise alat)
            margin = int(len(full_signal) * 0.05)
            # Pastikan sinyal cukup panjang untuk dipotong
            if len(full_signal) > target_len * 2:
                roi_signal = full_signal[margin:-margin]
                offset = margin
            else:
                roi_signal = full_signal
                offset = 0

            if len(roi_signal) == 0: roi_signal = full_signal

            # 2. Hitung Turunan (Differentiate)
            # Ini akan menonjolkan perubahan tajam (QRS complex) dan meredam gelombang landai (baseline wander)
            diff = np.diff(roi_signal)
            
            # 3. Kuadratkan (Squaring)
            # Membuat semua nilai positif dan memperbesar sinyal QRS yang kuat
            sq = diff ** 2
            
            # 4. Moving Window Integration (Smoothing energi)
            # Window kira-kira selebar QRS complex (misal 10-15 sampel)
            window_size = 15
            integrated = np.convolve(sq, np.ones(window_size)/window_size, mode='same')
            
            # 5. Cari Puncak Energi Tertinggi
            # Ini adalah lokasi QRS complex yang paling "jelas", bukan noise tinggi yang statis.
            peak_idx_roi = np.argmax(integrated)
            peak_idx = peak_idx_roi + offset
            
            # 6. Slicing Window di sekitar Puncak
            half_len = target_len // 2
            start_idx = peak_idx - half_len
            end_idx = peak_idx + half_len + 1 # +1 untuk handling ganjil
            
            # 7. Handle Boundary (Padding jika di ujung)
            pad_left = 0
            pad_right = 0
            
            if start_idx < 0:
                pad_left = abs(start_idx)
                start_idx = 0
            
            if end_idx > len(full_signal):
                pad_right = end_idx - len(full_signal)
                end_idx = len(full_signal)
            
            # Ambil potongan dari sinyal ASLI (bukan yang sudah diproses diff/sq)
            beat_segment = full_signal[start_idx:end_idx]
            
            if pad_left > 0 or pad_right > 0:
                beat_segment = np.pad(beat_segment, (pad_left, pad_right), 'edge')
                
            # Pastikan panjang pas 187
            if len(beat_segment) > target_len:
                beat_segment = beat_segment[:target_len]
            elif len(beat_segment) < target_len:
                beat_segment = np.pad(beat_segment, (0, target_len - len(beat_segment)), 'edge')
                
            return beat_segment
            
        except Exception as e:
            print(f"[Slicing Error] {e}, falling back to resize.")
            x_old = np.linspace(0, 1, len(full_signal))
            x_new = np.linspace(0, 1, target_len)
            return 
        np.interp(x_new, x_old, full_signal)

    def create_ecg_grid_plot(self, signal, rows=1, show_grid=True, show_details=False, filename="ECG Data"):
        try:
            rows = int(rows)
            points_per_row = len(signal) // rows
            fig, axes = plt.subplots(rows, 1, figsize=(15, 3 * rows), sharex=False, sharey=False)
            
            if rows == 1: axes = [axes]
            
            if show_details:
                fig.suptitle(f"ECG Report | Source: {filename} | Date: {np.datetime64('today')}", fontsize=14, fontweight='bold', y=0.98)
            
            for i, ax in enumerate(axes):
                start = i * points_per_row
                end = start + points_per_row
                segment = signal[start:end]
                
                ax.plot(segment, color='black', linewidth=1)
                
                if show_grid:
                    ax.minorticks_on()
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(which='major', linestyle='-', linewidth=1.2, color='#f0a1a1')
                    ax.grid(which='minor', linestyle='-', linewidth=0.5, color='#fce0e0')
                else:
                    ax.grid(False)
                    ax.axis('off')

                for spine in ax.spines.values(): spine.set_visible(False)
                ax.set_ylabel(f"Lead/Row {i+1}", fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[ECG Plotting Error] {e}")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    def predict_from_file(self, file_bytes, filename, options):
        if not self.model:
            return "Model Error", 0.0, "Gagal memuat model.", None

        # 1. Parse File
        full_signal = self.parse_file_to_signal(file_bytes, filename)
        if full_signal is None or len(full_signal) == 0:
            return "File Error", 0.0, "Gagal membaca format file.", None

        # 2. Generate Plot
        rows = options.get('rows', 1)
        grid = options.get('grid', 'true') == 'true'
        details = options.get('details', 'false') == 'true'
        ecg_plot_image = self.create_ecg_grid_plot(full_signal, rows, grid, details, filename)

        # 3. Preprocess untuk AI: Gunakan fungsi extraction yang baru
        signal_processed = self.extract_heartbeat(full_signal, target_len=187)

        # Normalize 0-1
        signal_processed = (signal_processed - np.min(signal_processed)) / (np.max(signal_processed) - np.min(signal_processed) + 1e-6)
        
        input_tensor = torch.tensor(signal_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # 4. Prediksi
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf_score, preds = torch.max(probs, 1)
            
            pred_idx = preds.item()
            label = self.classes_map.get(pred_idx, "Unknown")
            confidence = conf_score.item() * 100
            
            explanation = ""
            if pred_idx == 0: explanation = "Detak jantung normal. Pola gelombang P-QRS-T teratur dan stabil."
            elif pred_idx == 1: explanation = "Indikasi Supraventricular (S). Irama cepat abnormal yang berasal dari serambi jantung (atrium)."
            elif pred_idx == 2: explanation = "Indikasi Ventricular Ectopic (V). Kontraksi prematur pada bilik jantung, sering disebut extrasystole."
            elif pred_idx == 3: explanation = "Fusion Beat. Gabungan impuls listrik normal dan abnormal secara bersamaan."
            else: explanation = "Pola detak jantung cukup normal."

            return label, confidence, explanation, ecg_plot_image