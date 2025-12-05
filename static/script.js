// State
let batchQueue = [];
let batchResults = [];
let batchId = null;
let currentECGFile = null;
// Variabel tambahan untuk menyimpan pilihan user sebelum buka kamera
let lastSelectedType = 'bone'; 

// =================================================================
// 1. KAMUS DATA MEDIS (UNTUK UI)
// =================================================================
const fractureExplanations = {
    "Comminuted": { description: "Tulang hancur menjadi beberapa serpihan.", treatment: "Operasi (ORIF) pemasangan plat & sekrup.", healing: "3-6 bulan.", severity: "SANGAT SERIUS" },
    "Greenstick": { description: "Tulang retak/bengkok, umum pada anak-anak.", treatment: "Gips/Cast selama beberapa minggu.", healing: "4-6 minggu.", severity: "RINGAN" },
    "Healthy": { description: "Struktur tulang tampak normal dan utuh.", treatment: "Tidak diperlukan.", healing: "-", severity: "NORMAL" },
    "Linear": { description: "Patah tulang sejajar dengan poros tulang.", treatment: "Imobilisasi dengan gips.", healing: "6-8 minggu.", severity: "SEDANG" },
    "Oblique": { description: "Patah tulang miring/diagonal.", treatment: "Gips atau Operasi jika bergeser.", healing: "2-3 bulan.", severity: "SEDANG" },
    "Oblique Displaced": { description: "Patah miring dan tulang bergeser dari posisi.", treatment: "Operasi segera untuk reposisi.", healing: "3-4 bulan.", severity: "SERIUS" },
    "Transverse": { description: "Patah melintang tegak lurus.", treatment: "Gips atau Operasi.", healing: "2-3 bulan.", severity: "SEDANG" },
    "Transverse Displaced": { description: "Patah melintang & bergeser, celah antar tulang.", treatment: "Operasi fiksasi internal.", healing: "3-5 bulan.", severity: "SERIUS" },
    "Spiral": { description: "Patah melingkar akibat gerakan memutar.", treatment: "Imobilisasi.", healing: "2-3 bulan.", severity: "SEDANG" },
    "Fracture": { description: "Terdeteksi adanya fraktur pada struktur tulang.", treatment: "Konsultasi Ortopedi segera.", healing: "Bervariasi.", severity: "SERIUS" }
};

const skinExplanations = {
    "Scabies": {
        title: "Terindikasi Scabies",
        description: "Terdeteksi tanda-tanda Scabies atau kelainan kulit yang signifikan berdasarkan analisis AI. Segera konsultasikan ke dokter.",
        treatment: "Krim Permethrin 5%, Ivermectin, cuci baju air panas.",
        recommendation: "Segera berobat ke dokter kulit. Isolasi pakaian/handuk agar tidak menular."
    },
    "Healthy Skin": {
        title: "Kulit Sehat",
        description: "Tidak ditemukan tanda-tanda Scabies atau kelainan kulit yang signifikan berdasarkan analisis AI. Tetap jaga kebersihan kulit Anda.",
        treatment: "Perawatan kulit rutin.",
        recommendation: "Lanjutkan pola hidup bersih dan sehat. Lakukan pemeriksaan rutin jika ada keluhan."
    }
};

// =================================================================
// 2. HELPER UI GENERATORS (Layout Persis Screenshot)
// =================================================================

function generateBoneResultHTML(data) {
    const label = data.label || "Unknown";
    let infoKey = Object.keys(fractureExplanations).find(k => label.includes(k));
    if (!infoKey && label.toLowerCase().includes("fracture")) infoKey = "Fracture";
    const info = fractureExplanations[infoKey] || fractureExplanations["Healthy"];
    
    const isNormal = label.toLowerCase().includes("healthy");
    const colorClass = isNormal ? "text-green-600" : "text-red-600";
    const bgInfo = isNormal ? "#F0FDF4" : "#FEF2F2"; 
    const borderColor = isNormal ? "#BBF7D0" : "#FECACA"; 
    const severityColor = isNormal ? "#16a34a" : "#dc2626";

    const heatmapSrc = data.gradcam_image || data.original_image;
    const enhancedSrc = data.enhanced_image || data.original_image;

    let predsHTML = `<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">`;
    if (data.all_predictions) {
        for (const [k, v] of Object.entries(data.all_predictions)) {
            const isSelected = k === label;
            const style = isSelected ? `border-2 border-blue-500 bg-blue-50` : `border border-gray-200`;
            predsHTML += `
                <div class="p-3 rounded-lg ${style}">
                    <div class="text-xs font-bold text-gray-700">${k}</div>
                    <div class="text-xs text-gray-500">${v}%</div>
                </div>`;
        }
    }
    predsHTML += `</div>`;

    return `
        <!-- HEADER HASIL -->
        <div class="mb-4 p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
            <h4 class="${colorClass} text-xl font-bold mb-1">${label}</h4>
            <p class="text-gray-500 text-sm">Confidence: ${parseFloat(data.confidence).toFixed(2)}%</p>
        </div>

        <!-- 1. ANALISIS CITRA (ORIGINAL VS ENHANCED) -->
        <div class="mb-4 p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
            <div class="flex items-center gap-2 mb-4 font-bold text-gray-700">
                <i data-feather="image"></i> Analisis Citra (Super Resolution)
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div class="text-center">
                    <p class="text-xs text-gray-500 mb-2">Original X-Ray</p>
                    <img src="${data.original_image}" class="rounded-lg w-full h-48 object-contain border border-gray-200 bg-black">
                </div>
                <div class="text-center">
                    <p class="text-xs text-blue-500 mb-2 font-bold">Enhanced (Real-ESRGAN)</p>
                    <img src="${enhancedSrc}" class="rounded-lg w-full h-48 object-contain border-2 border-blue-400 bg-black">
                </div>
            </div>
            <p class="text-xs text-gray-400 mt-3 text-center italic">Citra ditingkatkan menggunakan AI GAN untuk memperjelas struktur retakan halus.</p>
        </div>

        <!-- 2. PENJELASAN MEDIS -->
        <div class="mb-4 p-5 rounded-xl border" style="background-color:${bgInfo}; border-color:${borderColor};">
            <h5 class="font-bold mb-4 text-gray-800">Penjelasan Medis</h5>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div>
                    <strong class="block text-gray-900 mb-1">Deskripsi:</strong>
                    <p class="text-gray-700">${info.description}</p>
                </div>
                <div>
                    <strong class="block text-gray-900 mb-1">Penanganan:</strong>
                    <p class="text-gray-700">${info.treatment}</p>
                </div>
            </div>
            <div class="mt-4 pt-4 border-t border-gray-300/50 flex justify-between items-center text-sm">
                <span><strong>Penyembuhan:</strong> ${info.healing}</span>
                <span class="px-3 py-1 rounded-full text-white text-xs font-bold shadow-sm" style="background-color:${severityColor}">${info.severity}</span>
            </div>
        </div>

        <!-- 3. CARD HEATMAP -->
        <div class="mb-4 p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
            <div class="flex items-center gap-2 mb-4 font-bold text-gray-700">
                <i data-feather="map"></i> Heatmap Area Masalah (GradCAM)
            </div>
            <div class="flex justify-center">
                <div class="relative">
                    <img src="${heatmapSrc}" class="rounded-lg max-h-64 object-contain border border-gray-200">
                    <p class="text-xs text-center text-gray-400 mt-2">Area merah menunjukkan fokus deteksi AI.</p>
                </div>
            </div>
        </div>

        <!-- 4. CARD PROBABILITAS -->
        <div class="p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
            <div class="flex items-center gap-2 mb-3 font-bold text-gray-700">
                <i data-feather="bar-chart-2"></i> Detail Probabilitas
            </div>
            ${predsHTML}
        </div>
    `;
}

function generateSkinResultHTML(data) {
    const label = data.label;
    const isScabies = label.toLowerCase().includes("scabies");
    const isHealthy = label.toLowerCase().includes("healthy");
    
    const info = isScabies ? skinExplanations["Scabies"] : skinExplanations["Healthy Skin"];
    
    const titleColor = isHealthy ? "text-green-500" : "text-red-500";
    const healthBoxClass = isHealthy ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200";
    const recBoxClass = "bg-green-50 border-green-200"; 
    const discBoxClass = "bg-yellow-50 border-yellow-200";

    const imgEnhanced = data.enhanced_image || data.original_image;
    
    return `
        <!-- 1. PERBANDINGAN GAMBAR -->
        <div class="mb-4 p-5 rounded-xl bg-white border border-gray-200 shadow-sm">
            <div class="flex items-center gap-2 mb-4 font-bold text-gray-700">
                <i data-feather="image"></i> Perbandingan Gambar
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div class="text-center">
                    <p class="text-xs text-gray-500 mb-2">Gambar Asli</p>
                    <img src="${data.original_image}" class="rounded-lg w-full h-48 object-cover border border-gray-200">
                </div>
                <div class="text-center">
                    <p class="text-xs text-blue-500 mb-2 font-bold">Gambar Enhanced (OpenCV)</p>
                    <img src="${imgEnhanced}" class="rounded-lg w-full h-48 object-cover border-2 border-blue-400">
                </div>
            </div>
            <p class="text-xs text-gray-400 mt-3 text-center italic">Note: Analisis dilakukan pada gambar yang telah ditingkatkan kualitasnya menggunakan OpenCV</p>
        </div>

        <!-- 2. DIAGNOSIS UTAMA -->
        <div class="mb-4 p-5 rounded-xl bg-white border border-gray-200 shadow-sm">
            <div class="flex items-center gap-2 mb-1 text-gray-700 font-bold">
                <i data-feather="check-circle"></i> Diagnosis Utama
            </div>
            <h2 class="text-2xl font-bold ${titleColor} mb-1">${label}</h2>
            <p class="text-sm text-gray-500">Tingkat Kepercayaan: ${parseFloat(data.confidence).toFixed(2)}%</p>
        </div>

        <!-- 3. STATUS KESEHATAN -->
        <div class="mb-4 p-5 rounded-xl border ${healthBoxClass}">
            <div class="flex items-center gap-2 mb-2 font-bold text-gray-700">
                <i data-feather="smile"></i> Status Kesehatan
            </div>
            <p class="text-sm text-gray-700 leading-relaxed">
                <strong>${info.title}.</strong> ${info.description}
            </p>
        </div>

        <!-- 4. REKOMENDASI & DISCLAIMER -->
        <div class="p-5 rounded-xl bg-white border border-gray-200 shadow-sm">
            <div class="flex items-center gap-2 mb-4 font-bold text-gray-700">
                <i data-feather="alert-circle"></i> Rekomendasi & Disclaimer
            </div>
            
            <div class="mb-3 p-4 rounded-lg border ${recBoxClass}">
                <strong class="text-green-800 text-xs uppercase block mb-1">REKOMENDASI:</strong>
                <p class="text-green-800 text-sm font-medium">${info.recommendation}</p>
            </div>

            <div class="p-4 rounded-lg border ${discBoxClass}">
                <strong class="text-yellow-800 text-xs uppercase block mb-1">DISCLAIMER:</strong>
                <p class="text-yellow-800 text-xs">Hasil ini dihasilkan oleh kecerdasan buatan (AI) dan hanya berfungsi sebagai alat bantu skrining awal, bukan pengganti diagnosis medis profesional.</p>
            </div>
        </div>
    `;
}

// =================================================================
// 3. MAIN LOGIC
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    feather.replace();
    setupDropzones();
    setupECGDropzone();
    const select = document.getElementById('singleType');
    if(select) select.addEventListener('change', (e) => updateInputAccept(e.target.value));
});

function updateInputAccept(type) {
    const input = document.getElementById('singleInput');
    const webcamBtn = document.getElementById('webcamBtn');
    if (type === 'ecg') { input.setAttribute('accept', '.ecg,.txt,.csv'); } 
    else { input.setAttribute('accept', 'image/*,.dcm'); }
}

function showPage(id) { document.querySelectorAll('.page-section').forEach(el => el.classList.remove('active')); document.getElementById(id).classList.add('active'); window.scrollTo(0,0); }
function showHome() { showPage('home-section'); }
function openSingleModal(t) { const m = document.getElementById('singleModal'); m.classList.remove('hidden'); m.classList.add('flex'); document.getElementById('singleType').value = t; updateInputAccept(t); }
function openECGModal() { document.getElementById('ecgModal').classList.remove('hidden'); document.getElementById('ecgModal').classList.add('flex'); }
function closeModal(id) { 
    document.getElementById(id).classList.add('hidden'); document.getElementById(id).classList.remove('flex'); 
    if(id==='liveSkinModal') stopLiveCamera();
}

function setupDropzones() {
    const zone = document.getElementById('single-dropzone');
    const input = document.getElementById('singleInput');
    const browseBtn = document.getElementById('browseBtn');
    const webcamBtn = document.getElementById('webcamBtn');
    if (!input || !zone) return;
    input.onchange = (e) => { if (e.target.files && e.target.files[0]) handleSingleFile(e.target.files[0]); };
    if (browseBtn) browseBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); input.click(); });
    if (webcamBtn) webcamBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); openCameraModal(); });
    zone.addEventListener('click', (e) => { if (e.target !== browseBtn && e.target !== webcamBtn && !browseBtn.contains(e.target) && !webcamBtn.contains(e.target)) { input.click(); } });
    zone.ondragover = (e) => { e.preventDefault(); zone.classList.add('border-blue-500'); };
    zone.ondragleave = () => { zone.classList.remove('border-blue-500'); };
    zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove('border-blue-500'); if (e.dataTransfer.files[0]) handleSingleFile(e.dataTransfer.files[0]); };
    
    // [RESTORED] Batch Input Handler
    const batchInput = document.getElementById('batchInput'); 
    if (batchInput) batchInput.onchange = (e) => runBatchProcess(e.target.files);
}

function setupECGDropzone() {
    const zone = document.getElementById('ecg-dropzone');
    const input = document.getElementById('ecgInput');
    if (!input || !zone) return;
    input.onchange = (e) => { if (e.target.files[0]) handleECGFile(e.target.files[0]); };
    zone.ondragover = (e) => { e.preventDefault(); zone.classList.add('border-red-500'); zone.classList.add('bg-red-100'); };
    zone.ondragleave = () => { zone.classList.remove('border-red-500'); zone.classList.remove('bg-red-100'); };
    zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove('border-red-500'); zone.classList.remove('bg-red-100'); if (e.dataTransfer.files[0]) handleECGFile(e.dataTransfer.files[0]); };
}

function handleSingleFile(file) {
    if(!file) return; window.singleFile = file; 
    const preview = document.getElementById('singlePreview');
    const content = document.getElementById('singlePreviewContent');
    preview.classList.remove('hidden'); preview.style.display = 'block'; 
    const type = document.getElementById('singleType').value;
    if(file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => content.innerHTML = `<img src="${e.target.result}" style="max-height:200px; margin:0 auto; border-radius:6px;">`;
        reader.readAsDataURL(file);
    } else {
        let icon = type === 'ecg' ? 'activity' : 'file-text';
        content.innerHTML = `<div class="p-6 bg-gray-50 text-gray-700 rounded border border-gray-200 flex flex-col items-center gap-2"><i data-feather="${icon}" class="w-10 h-10 text-blue-500"></i><span class="font-bold">${file.name}</span><span class="text-xs text-gray-500">Ready to analyze</span></div>`; feather.replace();
    }
}
function clearSinglePreview() { window.singleFile = null; document.getElementById('singlePreview').classList.add('hidden'); document.getElementById('singleInput').value = ''; document.getElementById('singlePreviewContent').innerHTML = ''; }
function handleECGFile(file) { if(!file) return; currentECGFile = file; document.getElementById('ecgPreview').classList.remove('hidden'); document.getElementById('ecgFilename').textContent = file.name; }

async function processSingle() {
    if(!window.singleFile) return alert("Please select a file.");
    const type = document.getElementById('singleType').value;
    const btn = document.querySelector('#singlePreview .analyze-btn');
    btn.innerHTML = 'Processing...'; btn.disabled = true;
    try {
        const fd = new FormData(); fd.append('file', window.singleFile); fd.append('type', type);
        const res = await fetch('/process-image', { method: 'POST', body: fd });
        if (!res.ok) throw new Error("Server Error");
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        closeModal('singleModal'); showResultModal(data); 
    } catch(e) { alert("Error: " + e.message); } finally { btn.innerHTML = 'Analyze Now'; btn.disabled = false; }
}

async function processECG() {
    if(!currentECGFile) return alert("Please select an ECG file.");
    const btn = document.getElementById('ecgAnalyzeBtn');
    btn.innerHTML = 'Processing...'; btn.disabled = true;
    const rows = document.getElementById('ecgRows').value;
    const grid = document.getElementById('ecgGrid').checked;
    const details = document.getElementById('ecgDetails').checked;
    const trim = document.getElementById('ecgTrim').checked;
    const windowSec = document.getElementById('ecgWindow').value;
    try {
        const fd = new FormData(); fd.append('file', currentECGFile); fd.append('type', 'ecg'); 
        fd.append('ecg_rows', rows); fd.append('ecg_grid', grid); 
        fd.append('ecg_details', details); fd.append('ecg_trim', trim); fd.append('ecg_window', windowSec);
        const res = await fetch('/process-image', { method: 'POST', body: fd });
        if (!res.ok) throw new Error("Server Error");
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        closeModal('ecgModal'); showResultModal(data); 
    } catch(e) { alert("ECG Error: " + e.message); } finally { btn.innerHTML = 'Analyze'; btn.disabled = false; }
}

function showResultModal(data) {
    let content = '';
    
    if(data.type === 'bone') {
        content = generateBoneResultHTML(data);
    } else if (data.type === 'skin') {
        content = generateSkinResultHTML(data);
    } else if (data.type === 'brain') {
        // [UPDATE] Brain UI juga disamakan strukturnya (Original vs Detection)
        const label = data.label;
        const isSafe = label.toLowerCase().includes('no tumor');
        const colorClass = isSafe ? "text-green-600" : "text-red-600";
        
        content = `
            <!-- HEADER -->
            <div class="mb-4 p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
                <h4 class="${colorClass} text-xl font-bold mb-1">${label}</h4>
                <p class="text-gray-500 text-sm">Confidence: ${parseFloat(data.confidence).toFixed(2)}%</p>
            </div>

            <!-- COMPARISON (Detection vs Mask) -->
            <div class="mb-4 p-5 rounded-xl bg-white border border-gray-100 shadow-sm">
                <div class="flex items-center gap-2 mb-4 font-bold text-gray-700">
                    <i data-feather="cpu"></i> Deteksi & Segmentasi
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div class="text-center">
                        <p class="text-xs text-gray-500 mb-2">Deteksi Tumor</p>
                        <img src="${data.annotated_image}" class="rounded-lg border border-gray-200 w-full object-cover">
                    </div>
                    ${data.mask_image ? `
                    <div class="text-center">
                        <p class="text-xs text-purple-500 mb-2 font-bold">Segmentasi Mask</p>
                        <img src="${data.mask_image}" class="rounded-lg border-2 border-purple-400 w-full object-cover bg-black">
                    </div>` : ''}
                </div>
            </div>

            <!-- DETAILS -->
            <div class="p-5 rounded-xl bg-gray-50 border border-gray-200">
                <h5 class="font-bold text-gray-800 mb-2">Analisis Medis</h5>
                <p class="text-sm text-gray-700 mb-1"><strong>Ukuran Tumor (Estimasi):</strong> ${data.tumor_size}</p>
                <p class="text-sm text-gray-600 italic">"${data.explanation}"</p>
            </div>
        `;
    } else if (data.type === 'ecg') {
        const isNormal = data.label.toLowerCase().includes('normal');
        const colorClass = isNormal ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50';
        content = `
            <div class="result-layout">
                <div class="mb-4 text-center bg-white p-2 rounded border shadow-inner">
                    <img src="${data.original_image}" class="mx-auto max-h-[400px] object-contain">
                </div>
                <div class="p-6 rounded-xl border ${colorClass} text-center">
                    <h4 class="text-2xl font-bold mb-1">${data.label}</h4>
                    <p class="font-mono text-sm opacity-80 mb-4">Confidence: ${data.confidence.toFixed(2)}%</p>
                    <div class="bg-white/60 p-4 rounded text-left"><p class="text-sm">${data.explanation}</p></div>
                </div>
            </div>`;
    }

    const modalHTML = `
        <div class="modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" id="tempResultModal">
            <div class="modal-content large bg-white rounded-lg p-6 w-full max-w-4xl relative max-h-[90vh] overflow-hidden flex flex-col">
                <div class="modal-header flex justify-between items-center mb-4 shrink-0">
                    <h3 class="text-xl font-bold">Analysis Result</h3>
                    <button onclick="document.getElementById('tempResultModal').remove()" class="modal-close hover:text-red-500"><i data-feather="x"></i></button>
                </div>
                <div class="modal-body bg-slate-50 p-4 rounded overflow-y-auto grow">
                    ${data.viewer_url ? `<a href="${data.viewer_url}" target="_blank" class="text-blue-600 underline mb-2 block">Open Viewer</a>` : ''}
                    ${content}
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    feather.replace();
}

// ... (Batch) ...
function startBatchFlow() {
    document.getElementById('batchInput').click();
}

async function runBatchProcess(files) {
    if(files.length === 0) return;
    
    batchQueue = Array.from(files);
    batchResults = [];
    batchId = "BATCH_" + Date.now();
    
    showPage('batch-loading-section');
    document.getElementById('verbose-logs').innerHTML = '<div class="log-line">> Initializing Batch Engine...</div>';
    document.getElementById('batch-grid').innerHTML = '';
    
    let processed = 0;
    
    for(const file of batchQueue) {
        processed++;
        updateBatchUI(processed, batchQueue.length, file.name);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('type', 'brain'); 
            formData.append('batch_id', batchId);
            
            addLog(`Uploading ${file.name} to Orthanc...`);
            
            const res = await fetch('/process-image', { method: 'POST', body: formData });
            
            if (!res.ok) throw new Error(`Server Error ${res.status}`);
            
            const data = await res.json();
            
            if(!data.error) {
                addLog(`Analysis Complete: ${data.label} (${data.confidence.toFixed(1)}%)`);
                batchResults.push(data);
                addBatchCard(data);
            } else {
                addLog(`Error analyzing ${file.name}: ${data.error}`, 'red');
            }
        } catch(e) {
            addLog(`Network Error on ${file.name}: ${e.message}`, 'red');
        }
    }
    
    addLog("Batch Processing Complete.", "#10b981");
    setTimeout(() => {
        showPage('batch-results-section');
        if(batchResults.length > 0 && batchResults[0].study_uid) {
            const btn = document.getElementById('batch-viewer-btn');
            btn.classList.remove('hidden');
            btn.onclick = () => window.open(batchResults[0].viewer_url, '_blank');
        }
    }, 1000);
}

function updateBatchUI(current, total, filename) {
    const pct = (current / total) * 100;
    document.getElementById('batch-progress-bar').style.width = pct + '%';
    addLog(`Processing File ${current}/${total}: ${filename}...`);
}

function addLog(text, color='#10b981') {
    const box = document.getElementById('verbose-logs');
    const line = document.createElement('div');
    line.className = 'log-line';
    line.style.color = color;
    line.innerText = `> ${text}`;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
}

function addBatchCard(data) {
    const grid = document.getElementById('batch-grid');
    const isSafe = data.label.toLowerCase().includes('healthy') || data.label.toLowerCase().includes('no tumor');
    
    const card = document.createElement('div');
    card.className = 'batch-card border rounded p-2 text-center bg-white shadow'; 
    card.onclick = () => showResultModal(data);
    
    let img = data.annotated_image || data.original_image;
    
    card.innerHTML = `
        <img src="${img}" class="batch-img w-full h-32 object-cover rounded mb-2">
        <div class="batch-info">
            <div class="batch-filename text-xs text-gray-500 truncate">${data.filename}</div>
            <span class="badge px-2 py-1 text-xs rounded text-white ${isSafe ? 'bg-green-500' : 'bg-red-500'}">${data.label}</span>
        </div>
    `;
    grid.appendChild(card);
}

// --- CAMERA UTILS (DEBUGGED & ENHANCED) ---
let liveStream = null;

function openLiveSkinModal() {
    const modal = document.getElementById('liveSkinModal');
    modal.classList.remove('hidden'); modal.classList.add('flex');
    loadLiveCameras();
    startLiveCamera();
}

async function loadLiveCameras() {
    const select = document.getElementById('liveCameraSelect');
    select.innerHTML = '<option value="">Default Camera</option>';
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        cameras.forEach(cam => {
            const opt = document.createElement('option');
            opt.value = cam.deviceId;
            opt.text = cam.label || `Camera ${select.length}`;
            select.appendChild(opt);
        });
    } catch(e) { console.error(e); }
}

async function startLiveCamera(deviceId = "") {
    const video = document.getElementById('liveVideo');
    const canvas = document.getElementById('liveCanvas');
    if(liveStream) { liveStream.getTracks().forEach(t=>t.stop()); }
    
    // Reset UI
    video.classList.remove('hidden');
    canvas.classList.add('hidden');
    document.getElementById('liveResultPlaceholder').classList.remove('hidden');
    document.getElementById('liveResultContainer').classList.add('hidden');
    document.getElementById('liveResultContainer').innerHTML = "";
    document.getElementById('liveActionBtns').classList.add('hidden');
    document.getElementById('liveControls').classList.remove('hidden'); // Show capture btn

    const constraints = { video: deviceId ? { deviceId: { exact: deviceId } } : true };
    try {
        liveStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = liveStream;
    } catch(e) { alert("Camera error: " + e.message); }
}

function stopLiveCamera() {
    if(liveStream) { liveStream.getTracks().forEach(t=>t.stop()); liveStream = null; }
}

function resetLiveCamera() {
    startLiveCamera(document.getElementById('liveCameraSelect').value);
}

function captureLiveSkin() {
    const video = document.getElementById('liveVideo');
    const canvas = document.getElementById('liveCanvas');
    
    if(video.videoWidth) {
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // UI Change
    video.classList.add('hidden');
    canvas.classList.remove('hidden');
    document.getElementById('liveControls').classList.add('hidden'); // Hide capture btn
    
    // Auto Analyze
    canvas.toBlob(analyzeLiveBlob, 'image/jpeg');
}

async function analyzeLiveBlob(blob) {
    const container = document.getElementById('liveResultContainer');
    const placeholder = document.getElementById('liveResultPlaceholder');
    
    // Tampilkan loading di panel sementara (sebagai feedback visual)
    placeholder.innerHTML = `<div class="animate-spin mb-2"><i data-feather="loader"></i></div><p>Menganalisis Kulit...</p>`;
    if (typeof feather !== 'undefined') feather.replace();

    const fd = new FormData();
    fd.append('file', blob);
    fd.append('type', 'skin'); // Force Type Skin

    try {
        const res = await fetch('/process-image', { method: 'POST', body: fd });
        if (!res.ok) throw new Error("Server Error");
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        
        // [MODIFIKASI] Tutup modal live dan buka modal hasil standar
        stopLiveCamera();
        closeModal('liveSkinModal');
        showResultModal(data);
        
        // Reset state modal live (untuk penggunaan berikutnya)
        setTimeout(() => {
             placeholder.innerHTML = 'Hasil analisis akan muncul di sini setelah capture.';
             placeholder.classList.remove('hidden');
             if(container) container.classList.add('hidden');
        }, 500);
        
    } catch(e) {
        placeholder.innerHTML = `<p class="text-red-500">Error: ${e.message}</p>`;
        document.getElementById('liveActionBtns').classList.remove('hidden');
    }
}

async function processCapturedPhoto(btnElement) {
    const canvas = document.getElementById('cameraCanvas');
    
    if(btnElement) {
        btnElement.innerHTML = `<div class="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div> Analyzing...`;
        btnElement.disabled = true;
    }
    
    // Force Type 'skin' per user request
    lastSelectedType = 'skin';

    canvas.toBlob(async (blob) => { 
        if (!blob) { alert("Capture failed"); return; }
        blob.name = "webcam.jpg"; 
        
        // Direct Process
        const formData = new FormData();
        formData.append('file', blob);
        formData.append('type', lastSelectedType);
        
        try {
            const res = await fetch('/process-image', { method: 'POST', body: formData });
            if (!res.ok) throw new Error("Server Error");
            const data = await res.json();
            if(data.error) throw new Error(data.error);
            
            stopCamera(); 
            closeModal('cameraModal'); 
            showResultModal(data); 
        } catch (e) {
            alert("Analysis Error: " + e.message);
            if(btnElement) {
                btnElement.innerHTML = `<i data-feather="activity" class="w-4 h-4 mr-2"></i> Analyze`;
                btnElement.disabled = false;
                feather.replace();
            }
        }
    }, 'image/jpeg');
}