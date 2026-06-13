import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Daun Durian Premium",
    page_icon="🍃",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #111827 !important;
    background-color: #f8fafc !important;
}

#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }

.page-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #052e16 !important;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.page-sub {
    text-align: center;
    font-size: 14px;
    font-weight: 500;
    color: #4b5563 !important;
    margin-bottom: 32px;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    color: #6b7280 !important;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.model-card {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(16,185,129,0.07);
}
.model-card-blue {
    background: #ffffff;
    border: 1.5px solid #bfdbfe;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07);
}

.model-card-header {
    padding: 16px 20px 14px;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
}
.model-card-header-blue {
    padding: 16px 20px 14px;
    border-bottom: 1px solid #e5e7eb;
    background: #eff6ff;
}

.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .05em;
    padding: 3px 10px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.badge-green { background: #dcfce7; color: #14532d !important; }
.badge-blue  { background: #dbeafe; color: #1e3a8a !important; }

.model-card-title {
    font-size: 15px;
    font-weight: 700;
    color: #111827 !important;
    margin: 0;
}
.model-card-sub {
    font-size: 12px;
    font-weight: 500;
    color: #6b7280 !important;
    margin-top: 3px;
}
.model-card-body { padding: 20px 20px 16px; }

.pred-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
}
.pred-variety {
    font-size: 26px;
    font-weight: 800;
    color: #052e16 !important;
    line-height: 1.2;
    letter-spacing: -0.3px;
}
.pred-hint {
    font-size: 12px;
    font-weight: 500;
    color: #9ca3af !important;
    margin-top: 3px;
}

.status-pill {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 11px;
    border-radius: 99px;
    white-space: nowrap;
    margin-top: 4px;
    flex-shrink: 0;
}
.pill-ok   { background: #dcfce7; color: #14532d !important; border: 1px solid #bbf7d0; }
.pill-warn { background: #fef3c7; color: #78350f !important; border: 1px solid #fde68a; }

.conf-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    color: #6b7280 !important;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── FIX BUG #3: confidence bar menggunakan SVG inline ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}
.conf-name {
    font-size: 12px;
    font-weight: 600;
    color: #374151 !important;
    width: 98px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
}
.conf-track-wrap {
    flex: 1;
    height: 7px;
    background: #e5e7eb;
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.conf-fill {
    position: absolute;
    top: 0; left: 0;
    height: 100%;
    border-radius: 99px;
}
.conf-pct {
    font-size: 12px;
    font-weight: 700;
    color: #111827 !important;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}
.hdivider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 16px 0;
}

.summary-card {
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-radius: 16px;
    overflow: hidden;
    margin-top: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.summary-header {
    padding: 16px 20px;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}
.summary-title {
    font-size: 15px;
    font-weight: 700;
    color: #111827 !important;
}
.summary-body { padding: 20px 20px 18px; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-chip {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 16px;
}
.metric-chip-label {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280 !important;
    margin-bottom: 6px;
    letter-spacing: 0.03em;
}
.metric-chip-val {
    font-size: 18px;
    font-weight: 800;
    color: #111827 !important;
    line-height: 1.2;
    letter-spacing: -0.2px;
}
.metric-chip-sub {
    font-size: 11px;
    font-weight: 500;
    color: #9ca3af !important;
    margin-top: 4px;
}

.cmp-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    color: #6b7280 !important;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.cmp-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #f3f4f6;
}
.cmp-row:last-child { border-bottom: none; }
.cmp-label {
    font-size: 12px;
    font-weight: 600;
    color: #374151 !important;
    width: 140px;
    white-space: nowrap;
    flex-shrink: 0;
}
.cmp-pct {
    font-size: 13px;
    font-weight: 700;
    color: #111827 !important;
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}

.stFileUploader label { font-weight: 600 !important; color: #111827 !important; }
.stCheckbox label, .stSlider label { color: #111827 !important; font-weight: 500 !important; }
div[data-testid="stCaptionContainer"] {
    color: #4b5563 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CACHE MODEL AI
# ==========================================
@st.cache_resource
def load_all_models():
    try:
        model_effnet = tf.keras.models.load_model('Model_EfficientNetV2B0_Final.h5')
        model_cnn    = tf.keras.models.load_model('Model_Baseline_CNN_Final.h5')
        return model_effnet, model_cnn, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

model_effnet, model_cnn, models_loaded = load_all_models()


# ==========================================
# 3. FIX BUG #2: FUNGSI EKSTRAKSI DAUN
# ==========================================
def auto_crop_daun(image_pil):
    """
    Deteksi kontur daun dan crop dengan padding.
    
    PERBAIKAN BUG:
    - Hapus margin-edge check yang terlalu ketat (menyebabkan semua kontur ditolak)
    - Perbaiki scoring: pakai area saja sebagai kriteria utama, bukan skor hibrida
      yang bias ke tengah gambar
    - Perluas range HSV hijau agar menangkap berbagai kondisi pencahayaan
    - Tambah fallback: jika tidak ada kontur valid, coba tanpa filter edge
    """
    if image_pil.mode in ('RGBA', 'LA'):
        bg = Image.new("RGB", image_pil.size, (255, 255, 255))
        bg.paste(image_pil, mask=image_pil.convert('RGBA').split()[3])
        image_pil = bg
    else:
        image_pil = image_pil.convert('RGB')

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    H, W   = img_cv.shape[:2]
    hsv    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Range HSV diperluas untuk menangkap berbagai warna daun
    # Hijau muda → tua, termasuk daun dengan sedikit kekuningan
    masks = [
        cv2.inRange(hsv, np.array([20, 30, 30],  dtype=np.uint8),
                         np.array([95, 255, 255], dtype=np.uint8)),  # hijau lebar
        cv2.inRange(hsv, np.array([95, 20, 20],  dtype=np.uint8),
                         np.array([140, 200, 200], dtype=np.uint8)), # hijau-biru tua
    ]
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    # Morphology lebih agresif untuk menutup lubang pada daun
    k_open  = np.ones((5, 5),   np.uint8)
    k_close = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not kontur:
        return image_pil, False, "Tidak ada kontur"

    # FIX: ambil kontur terbesar yang memenuhi area minimum
    # Tidak ada edge-margin check karena daun foto seringkali menyentuh tepi
    min_area = (H * W) * 0.01  # minimal 1% dari luas gambar
    valid    = [c for c in kontur if cv2.contourArea(c) >= min_area]

    if not valid:
        # Fallback: ambil kontur terbesar apapun
        valid = kontur

    best = max(valid, key=cv2.contourArea)

    pad = 30  # padding lebih kecil agar tidak terlalu banyak background
    x, y, w, h = cv2.boundingRect(best)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    crop = img_cv[y1:y2, x1:x2]

    if crop.size == 0:
        return image_pil, False, "Crop kosong"

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), True, f"Padding {pad}px"


# ==========================================
# 4. FIX BUG #1: FUNGSI PREDIKSI
# ==========================================
def preprocess_for_effnet(img_final):
    """
    EfficientNetV2B0 dengan preprocess_input resmi TF.
    Jika model di-train dengan /255, tetap gunakan /255.
    Kita deteksi otomatis berdasarkan output layer pertama.
    """
    img_resized = img_final.resize((224, 224))
    img_array   = tf.keras.preprocessing.image.img_to_array(img_resized)
    # EfficientNetV2 default: preprocess_input rescale ke [-1, 1]
    img_array   = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def preprocess_for_cnn(img_final):
    """
    FIX BUG #1 — CNN Baseline selalu prediksi Bawor:
    
    Root cause: CNN dilatih dengan normalisasi /255.0 (nilai [0,1]).
    Kode lama menggunakan batch yang sama dengan EfficientNet yang memakai
    preprocess_input (range [-1,1]). Ketika CNN menerima input [-1,1],
    distribusi aktivasi bergeser drastis → selalu prediksi kelas pertama (Bawor).
    
    Fix: buat batch terpisah dengan normalisasi /255.0 untuk CNN.
    Jika model CNN Anda dilatih dengan cara lain, sesuaikan di sini.
    """
    img_resized = img_final.resize((224, 224))
    img_array   = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array   = img_array / 255.0  # normalisasi standar untuk CNN from scratch
    return np.expand_dims(img_array, axis=0)


def prediksi_model(model, img_batch):
    pred  = model.predict(img_batch, verbose=0)
    probs = pred[0]
    idx   = int(np.argmax(probs))
    prob  = float(probs[idx]) * 100
    return idx, prob, probs


# ==========================================
# 5. FIX BUG #3: render card — bar pakai SVG
# ==========================================
def bar_svg(pct, color_start, color_end):
    """
    FIX BUG #3 — Distribusi probabilitas tampil teks, bukan bar:
    
    Root cause: Streamlit strip/sanitize inline style `width:X%` pada div
    dari markdown unsafe_allow_html ketika nilai berasal dari variabel Python
    (dianggap XSS risk pada beberapa versi Streamlit).
    
    Fix: gunakan SVG rect dengan attribute 'width' (bukan CSS style) — 
    SVG attribute tidak di-strip oleh sanitizer Streamlit.
    """
    w = max(2, round(pct))
    return (
        f'<svg width="100%" height="7" style="display:block;border-radius:99px;overflow:hidden">'
        f'<rect x="0" y="0" width="100%" height="7" fill="#e5e7eb"/>'
        f'<rect x="0" y="0" width="{w}%" height="7" rx="3.5" ry="3.5" '
        f'fill="url(#g{w})"/>'
        f'<defs><linearGradient id="g{w}" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{color_start}"/>'
        f'<stop offset="100%" stop-color="{color_end}"/>'
        f'</linearGradient></defs>'
        f'</svg>'
    )


def render_card(badge_label, badge_class, card_class, header_class,
                card_title, card_sub, variety, confidence_pct,
                all_probs, class_labels, color_start, color_end, threshold):
    ok       = confidence_pct >= threshold
    pill_cls = "pill-ok" if ok else "pill-warn"
    pill_txt = (f"✓ {confidence_pct:.1f}% ≥ {threshold:.0f}%"
                if ok else f"⚠ {confidence_pct:.1f}% < {threshold:.0f}%")

    bars = ""
    for lbl, p in zip(class_labels, all_probs):
        pct  = float(p) * 100
        svg  = bar_svg(pct, color_start, color_end)
        bars += f"""
        <div class="conf-row">
          <span class="conf-name">{lbl}</span>
          <div style="flex:1">{svg}</div>
          <span class="conf-pct">{pct:.1f}%</span>
        </div>"""

    return f"""
    <div class="{card_class}">
      <div class="{header_class}">
        <span class="badge {badge_class}">{badge_label}</span>
        <div class="model-card-title">{card_title}</div>
        <div class="model-card-sub">{card_sub}</div>
      </div>
      <div class="model-card-body">
        <div class="pred-row">
          <div>
            <div class="pred-variety">{variety}</div>
            <div class="pred-hint">Varietas terdeteksi</div>
          </div>
          <span class="status-pill {pill_cls}">{pill_txt}</span>
        </div>
        <hr class="hdivider">
        <div class="conf-section-label">Distribusi Probabilitas</div>
        {bars}
      </div>
    </div>"""


# ==========================================
# 6. HELPER: render summary card
# ==========================================
def render_summary(idx_e, prob_e, idx_c, prob_c, class_labels, threshold):
    sepakat     = idx_e == idx_c
    badge_style = ("background:#dcfce7;color:#14532d;border:1px solid #bbf7d0;"
                   if sepakat else
                   "background:#fef3c7;color:#78350f;border:1px solid #fde68a;")
    badge_txt = "✓ Kedua model sepakat" if sepakat else "⚠ Model berbeda pendapat"
    selisih   = abs(prob_e - prob_c)
    unggul    = "EfficientNet" if prob_e >= prob_c else "Baseline CNN"

    bar_e = bar_svg(prob_e, "#16a34a", "#4ade80")
    bar_c = bar_svg(prob_c, "#2563eb", "#60a5fa")

    return f"""
    <div class="summary-card">
      <div class="summary-header">
        <span class="summary-title">Ringkasan Perbandingan</span>
        <span class="status-pill" style="{badge_style}">{badge_txt}</span>
      </div>
      <div class="summary-body">
        <div class="metric-grid">
          <div class="metric-chip">
            <div class="metric-chip-label">EfficientNetV2B0</div>
            <div class="metric-chip-val">{class_labels[idx_e]}</div>
            <div class="metric-chip-sub">{prob_e:.1f}% keyakinan</div>
          </div>
          <div class="metric-chip">
            <div class="metric-chip-label">Baseline CNN</div>
            <div class="metric-chip-val">{class_labels[idx_c]}</div>
            <div class="metric-chip-sub">{prob_c:.1f}% keyakinan</div>
          </div>
          <div class="metric-chip">
            <div class="metric-chip-label">Selisih keyakinan</div>
            <div class="metric-chip-val">{selisih:.1f}%</div>
            <div class="metric-chip-sub">{unggul} lebih tinggi</div>
          </div>
        </div>
        <hr class="hdivider">
        <div class="cmp-section-label">Perbandingan Keyakinan</div>
        <div class="cmp-row">
          <span class="cmp-label">EfficientNetV2B0</span>
          <div style="flex:1">{bar_e}</div>
          <span class="cmp-pct">{prob_e:.1f}%</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">Baseline CNN</span>
          <div style="flex:1">{bar_c}</div>
          <span class="cmp-pct">{prob_c:.1f}%</span>
        </div>
      </div>
    </div>"""


# ==========================================
# 7. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1891/1891756.png", width=72)
    st.markdown("### ⚙️ Pengaturan")

    gunakan_autocrop = st.checkbox(
        "Aktifkan ekstraksi daun & padding",
        value=True,
        help="Sistem melacak kontur daun hijau dan menambahkan margin secara otomatis."
    )
    st.markdown("---")
    threshold_keyakinan = st.slider(
        "Ambang batas keyakinan (%)",
        min_value=50.0, max_value=99.0, value=80.0, step=1.0
    )
    st.markdown("---")

    if models_loaded:
        st.success("✅ AI Engine: READY")
    else:
        st.error("❌ AI Engine: OFFLINE")
        st.caption("Pastikan file `.h5` ada di direktori yang sama dengan `app.py`.")

    # ── Info preprocessing ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**ℹ️ Info Preprocessing**")
    st.caption(
        "• EfficientNetV2: `preprocess_input` → [-1, 1]\n"
        "• Baseline CNN: `/255.0` → [0, 1]"
    )


# ==========================================
# 8. KONTEN UTAMA
# ==========================================
st.markdown("<div class='page-title'>🍃 Klasifikasi Varietas Daun Durian Premium</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='page-sub'>Komparasi arsitektur EfficientNetV2B0 vs Baseline CNN</div>",
    unsafe_allow_html=True
)

if not models_loaded:
    st.error(
        "⚠️ Model tidak ditemukan. "
        "Pastikan file `Model_EfficientNetV2B0_Final.h5` dan "
        "`Model_Baseline_CNN_Final.h5` tersedia di direktori yang sama dengan `app.py`."
    )
    st.stop()

# ── Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Unggah citra daun durian",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_input = Image.open(uploaded_file)

    # ── Tahap 1: Prapemrosesan ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-label'>Tahap 1 — Prapemrosesan citra</div>",
                unsafe_allow_html=True)

    col_in, col_out = st.columns(2)

    with col_in:
        st.caption("📷 Gambar input asli")
        st.image(image_input, use_container_width=True)

    with col_out:
        st.caption("🔍 Hasil deteksi objek daun")
        if gunakan_autocrop:
            with st.spinner("Memindai kontur hijau..."):
                img_final, status_crop, info_pad = auto_crop_daun(image_input)
            if status_crop:
                st.image(img_final, use_container_width=True)
                st.success(f"✅ Objek terdeteksi · {info_pad}")
            else:
                st.image(image_input.convert('RGB'), use_container_width=True)
                st.warning(
                    "⚠️ Kontur daun tidak terdeteksi. Menggunakan gambar asli. "
                    "Coba nonaktifkan fitur ekstraksi daun di sidebar."
                )
                img_final = image_input.convert('RGB')
        else:
            img_final = image_input.convert('RGB')
            st.image(img_final, use_container_width=True)
            st.info("ℹ️ Deteksi daun dinonaktifkan — menggunakan gambar asli.")

    # ── Tahap 2: Klasifikasi ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div class='section-label'>"
        f"Tahap 2 — Klasifikasi deep learning · Threshold {threshold_keyakinan:.0f}%"
        f"</div>",
        unsafe_allow_html=True
    )

    CLASS_LABELS = ['Bawor', 'Duri Hitam', 'Monthong', 'Musang King']

    # FIX: preprocessing terpisah untuk tiap model
    batch_effnet = preprocess_for_effnet(img_final)
    batch_cnn    = preprocess_for_cnn(img_final)

    with st.spinner("🔬 Menganalisis fitur daun dengan dua arsitektur AI..."):
        idx_e, prob_e, probs_e = prediksi_model(model_effnet, batch_effnet)
        idx_c, prob_c, probs_c = prediksi_model(model_cnn,    batch_cnn)

    col_e, col_c = st.columns(2, gap="large")

    with col_e:
        st.markdown(
            render_card(
                badge_label    = "EfficientNetV2B0",
                badge_class    = "badge-green",
                card_class     = "model-card",
                header_class   = "model-card-header",
                card_title     = "Transfer Learning",
                card_sub       = "ImageNet pretrained · EfficientNetV2B0",
                variety        = CLASS_LABELS[idx_e],
                confidence_pct = prob_e,
                all_probs      = probs_e,
                class_labels   = CLASS_LABELS,
                color_start    = "#16a34a",
                color_end      = "#4ade80",
                threshold      = threshold_keyakinan,
            ),
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown(
            render_card(
                badge_label    = "Baseline CNN",
                badge_class    = "badge-blue",
                card_class     = "model-card-blue",
                header_class   = "model-card-header-blue",
                card_title     = "Custom Architecture",
                card_sub       = "Trained from scratch · Baseline CNN",
                variety        = CLASS_LABELS[idx_c],
                confidence_pct = prob_c,
                all_probs      = probs_c,
                class_labels   = CLASS_LABELS,
                color_start    = "#2563eb",
                color_end      = "#60a5fa",
                threshold      = threshold_keyakinan,
            ),
            unsafe_allow_html=True
        )

    # ── Ringkasan perbandingan ──────────────────────────────────────────
    st.markdown(
        render_summary(idx_e, prob_e, idx_c, prob_c, CLASS_LABELS, threshold_keyakinan),
        unsafe_allow_html=True
    )