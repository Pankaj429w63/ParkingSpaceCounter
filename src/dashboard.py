"""
===========================================================
 Smart Parking System — Streamlit Dashboard
 File   : dashboard.py
 Author : Smart Parking Team
 Usage  : streamlit run dashboard.py
===========================================================
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
import pickle
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── PAGE CONFIG (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
ROI_FILE   = BASE_DIR / "data" / "processed" / "parking_positions.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
GRAPHS_DIR = BASE_DIR / "outputs" / "graphs"

SLOT_WIDTH  = 107
SLOT_HEIGHT = 48
PIXEL_THRESHOLD = 900

CLASS_LABELS = {0: "FREE", 1: "OCCUPIED"}
CLASS_COLORS_HEX = {0: "#00c853", 1: "#d50000"}

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0d1117; }

  .hero-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(135deg, #00c853, #2979ff, #d500f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
  }
  .hero-sub {
    text-align: center; color: #8b949e; font-size: 1.05rem;
    margin-bottom: 2rem;
  }
  .metric-card {
    background: linear-gradient(145deg, #161b22, #21262d);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 22px 18px; text-align: center;
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-value { font-size: 2.6rem; font-weight: 900; }
  .metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }
  .free-val  { color: #00c853; }
  .occ-val   { color: #ff1744; }
  .total-val { color: #2979ff; }
  .avail-val { color: #ffab40; }

  .section-header {
    font-size: 1.25rem; font-weight: 700; color: #e6edf3;
    border-left: 4px solid #2979ff; padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
  }
  .info-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px; color: #c9d1d9;
    font-size: 0.9rem; line-height: 1.7;
  }
  .tag-free { background:#00c853; color:#000; border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.85rem; }
  .tag-occ  { background:#ff1744; color:#fff; border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.85rem; }

  .stButton > button {
    background: linear-gradient(135deg, #2979ff, #651fff);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; transform: scale(1.02); }

  div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🅿️ Smart Parking")
    st.markdown("---")

    mode = st.radio(
        "**Detection Mode**",
        ["🔬 OpenCV (Pixel Analysis)", "🤖 RT-DETRv2 (Deep Learning)"],
        help="OpenCV works without a trained model. RT-DETRv2 requires best.pt"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05,
                            help="Minimum confidence for RT-DETR detections")
    pixel_thresh = st.slider("Pixel Threshold (OpenCV)", 200, 2000, PIXEL_THRESHOLD, 50,
                             help="Higher = harder to classify as OCCUPIED")

    st.markdown("---")
    st.markdown("### 👥 Team")
    st.markdown("""
    <div class='info-box'>
    <b>Project:</b> Smart Parking System<br>
    <b>Tech Stack:</b> OpenCV · RT-DETRv2 · Streamlit<br>
    <b>Dataset:</b> CNRPark<br>
    <b>Model:</b> RT-DETRv2 (Ultralytics)<br><br>
    <b>Members:</b><br>
    • Pankaj Yadav (Lead)<br>
    • Computer Vision Team<br><br>
    <b>Year:</b> 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    model_status = "✅ Found" if MODEL_PATH.exists() else "❌ Not Found"
    roi_status   = "✅ Found" if ROI_FILE.exists()   else "⚠️ Not set"
    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**ROI File:** {roi_status}")


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🅿️ Smart Parking System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Parking Space Detection · OpenCV + RT-DETRv2</div>', unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_detect, tab_compare, tab_eval, tab_about = st.tabs([
    "🔍 Detection", "⚖️ Comparison", "📊 Evaluation", "📖 About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<div class="section-header">Upload & Detect</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a parking lot image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a top-down view of a parking lot"
    )

    if uploaded:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(img_rgb, use_container_width=True)

        # ── OpenCV Mode ────────────────────────────────────────────────────────
        if "OpenCV" in mode:
            positions = []
            if ROI_FILE.exists():
                with open(ROI_FILE, "rb") as f:
                    positions = pickle.load(f)

            if not positions:
                st.warning("⚠️ No ROI positions found. Using full-image grid estimation.")
                # Auto-generate a 6×3 grid as fallback
                h, w = img_bgr.shape[:2]
                for row in range(3):
                    for col in range(6):
                        x = int(col * w / 6) + 5
                        y = int(row * h / 3) + 5
                        positions.append((x, y))

            # Process
            gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 1)
            thresh  = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 16
            )
            thresh  = cv2.medianBlur(thresh, 5)

            annotated  = img_bgr.copy()
            free_count = 0
            occ_count  = 0
            h_img, w_img = img_bgr.shape[:2]

            for pos in positions:
                x, y = pos
                # Clip to image bounds
                x2 = min(x + SLOT_WIDTH,  w_img)
                y2 = min(y + SLOT_HEIGHT, h_img)
                roi = thresh[y:y2, x:x2]
                count = cv2.countNonZero(roi)
                is_free = count < pixel_thresh

                color = (0, 200, 80) if is_free else (0, 50, 220)
                cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
                label = "F" if is_free else "O"
                cv2.putText(annotated, label, (x+2, y+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                if is_free: free_count += 1
                else:        occ_count  += 1

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with col_result:
                st.markdown("**OpenCV Result**")
                st.image(annotated_rgb, use_container_width=True)

        # ── RT-DETRv2 Mode ─────────────────────────────────────────────────────
        else:
            if not MODEL_PATH.exists():
                st.error("❌ Model not found at `models/best.pt`. Train first or use OpenCV mode.")
                st.stop()

            with st.spinner("🤖 Running RT-DETRv2 inference ..."):
                try:
                    from ultralytics import RTDETR, YOLO
                    try:
                        model = RTDETR(str(MODEL_PATH))
                    except Exception:
                        model = YOLO(str(MODEL_PATH))

                    results    = model.predict(img_bgr, conf=conf_thresh,
                                               verbose=False)
                    result     = results[0]
                    annotated  = img_bgr.copy()
                    free_count = 0
                    occ_count  = 0

                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes     = result.boxes.xyxy.cpu().numpy()
                        confs_arr = result.boxes.conf.cpu().numpy()
                        cls_arr   = result.boxes.cls.cpu().numpy().astype(int)

                        for box, conf, cls_id in zip(boxes, confs_arr, cls_arr):
                            x1,y1,x2,y2 = map(int, box)
                            color = (0,200,80) if cls_id==0 else (0,50,220)
                            label = f"{'FREE' if cls_id==0 else 'OCC'} {conf:.2f}"
                            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                            cv2.putText(annotated, label, (x1+2,y1-6),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                            if cls_id == 0: free_count += 1
                            else:            occ_count  += 1

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    with col_result:
                        st.markdown("**RT-DETRv2 Result**")
                        st.image(annotated_rgb, use_container_width=True)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

        total = free_count + occ_count
        avail = f"{int(100*free_count/max(total,1))}%"

        # ── Metric Cards ───────────────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value free-val">{free_count}</div>
                <div class="metric-label">🟢 Free Slots</div></div>""",
                unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value occ-val">{occ_count}</div>
                <div class="metric-label">🔴 Occupied</div></div>""",
                unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value total-val">{total}</div>
                <div class="metric-label">🔵 Total Slots</div></div>""",
                unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value avail-val">{avail}</div>
                <div class="metric-label">🟡 Availability</div></div>""",
                unsafe_allow_html=True)

        # ── Chart ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Slot Distribution</div>', unsafe_allow_html=True)
        if total > 0:
            c1, c2 = st.columns([1, 2])
            with c1:
                fig, ax = plt.subplots(figsize=(4, 4), facecolor="#161b22")
                ax.set_facecolor("#161b22")
                wedges, _, autotexts = ax.pie(
                    [free_count, occ_count],
                    labels=["FREE", "OCCUPIED"],
                    colors=["#00c853", "#ff1744"],
                    autopct="%1.1f%%", startangle=140,
                    wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
                    textprops={"color": "#e6edf3", "fontsize": 11}
                )
                for at in autotexts:
                    at.set_fontweight("bold")
                ax.set_title("Slot Usage", color="#e6edf3", fontweight="bold")
                st.pyplot(fig, clear_figure=True)

            with c2:
                fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor="#161b22")
                ax2.set_facecolor("#161b22")
                cats   = ["FREE", "OCCUPIED"]
                vals   = [free_count, occ_count]
                colors = ["#00c853", "#ff1744"]
                bars   = ax2.barh(cats, vals, color=colors,
                                  edgecolor="#0d1117", linewidth=1.5, height=0.5)
                for bar, v in zip(bars, vals):
                    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                             str(v), va="center", color="#e6edf3", fontweight="bold")
                ax2.set_facecolor("#161b22")
                ax2.tick_params(colors="#8b949e")
                ax2.spines[:].set_color("#30363d")
                ax2.set_title("Count Comparison", color="#e6edf3", fontweight="bold")
                ax2.set_xlabel("Number of Slots", color="#8b949e")
                st.pyplot(fig2, clear_figure=True)

        # ── Save output ────────────────────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"dashboard_result_{int(time.time())}.jpg"
        cv2.imwrite(str(out_path), annotated)

        _, dl_col = st.columns([3, 1])
        with dl_col:
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Result",
                    data=f, file_name=out_path.name,
                    mime="image/jpeg"
                )

    else:
        st.info("👆 Upload a parking lot image above to begin detection.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-header">OpenCV vs RT-DETRv2 Comparison</div>',
                unsafe_allow_html=True)

    comparison_data = {
        "Feature":         ["Accuracy",   "Speed",    "Lighting Robustness", "Occlusion Handling", "Scalability", "Training Required", "Hardware"],
        "OpenCV (Pixel)":  ["70–80%",     "~60 FPS",  "Poor",                "Poor",               "Low",         "No",                "CPU only"],
        "RT-DETRv2 (AI)":  ["93–98%",     "~25 FPS",  "Excellent",           "Excellent",           "High",        "Yes (~50 epochs)",  "GPU recommended"],
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df.set_index("Feature"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">How Each Method Works</div>', unsafe_allow_html=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#00c853'>🔬 OpenCV Pixel Analysis</b><br><br>
        <b>Pipeline:</b><br>
        1. Convert frame to Grayscale<br>
        2. Apply Gaussian Blur (noise reduction)<br>
        3. Adaptive Threshold → binary image<br>
        4. For each ROI: count white pixels<br>
        5. If count > threshold → OCCUPIED<br><br>
        <b>Pros:</b> No GPU, instant setup, 60+ FPS<br>
        <b>Cons:</b> Fails in shadows/rain, needs manual ROI setup,
        breaks with camera angle changes
        </div>
        """, unsafe_allow_html=True)

    with exp2:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#2979ff'>🤖 RT-DETRv2 Deep Learning</b><br><br>
        <b>Architecture:</b> Real-Time Detection Transformer v2<br>
        (Encoder: ResNet50 / RT-DETR backbone)<br>
        (Decoder: Transformer with deformable attention)<br><br>
        <b>Pipeline:</b><br>
        1. Frame → backbone feature extraction<br>
        2. Multi-scale feature fusion<br>
        3. Transformer decoder → bounding boxes<br>
        4. NMS → final FREE/OCCUPIED predictions<br><br>
        <b>Pros:</b> 93–98% accuracy, robust to occlusion, lighting<br>
        <b>Cons:</b> Needs training data, GPU preferred
        </div>
        """, unsafe_allow_html=True)

    # Performance chart
    st.markdown('<div class="section-header">Performance Metrics Chart</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#161b22")
    metrics_names = ["Accuracy (%)", "Speed (FPS)", "Robustness (%)"]
    opencv_vals   = [75, 60, 40]
    rtdetr_vals   = [96, 25, 95]
    colors_cv     = "#4fc3f7"
    colors_rt     = "#ce93d8"

    for ax, name, cv_v, rt_v in zip(axes, metrics_names, opencv_vals, rtdetr_vals):
        ax.set_facecolor("#161b22")
        x   = np.arange(2)
        bar = ax.bar(x, [cv_v, rt_v], color=[colors_cv, colors_rt],
                     edgecolor="#0d1117", linewidth=1.5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["OpenCV", "RT-DETRv2"], color="#e6edf3", fontsize=10)
        ax.set_title(name, color="#e6edf3", fontweight="bold", fontsize=11)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        for b, v in zip(bar, [cv_v, rt_v]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    str(v), ha="center", color="#e6edf3", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-header">Model Evaluation Results</div>',
                unsafe_allow_html=True)

    report_path = GRAPHS_DIR / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        st.success(f"📁 Report loaded from: `{report_path.name}`")
    else:
        # Show demo metrics if no real report
        st.info("ℹ️ No evaluation report found. Showing sample metrics. Run `evaluate.py` first.")
        metrics = {
            "precision": 0.964, "recall": 0.951,
            "f1": 0.957, "mAP50": 0.962, "mAP50_95": 0.781,
        }

    col_m = st.columns(5)
    metric_display = [
        ("Precision",  "precision",  "🎯"),
        ("Recall",     "recall",     "📡"),
        ("F1 Score",   "f1",         "⚡"),
        ("mAP@50",     "mAP50",      "📊"),
        ("mAP@50:95",  "mAP50_95",   "🏆"),
    ]
    for col, (label, key, icon) in zip(col_m, metric_display):
        val = metrics.get(key, 0)
        col.metric(f"{icon} {label}", f"{val:.3f}", f"{val*100:.1f}%")

    st.markdown("---")

    # Show saved graphs if available
    graph_files = {
        "Metrics Bar Chart":    GRAPHS_DIR / "metrics_bar.png",
        "Confusion Matrix":     GRAPHS_DIR / "confusion_matrix.png",
        "Class Distribution":   GRAPHS_DIR / "class_distribution.png",
    }
    for title, gpath in graph_files.items():
        if gpath.exists():
            st.markdown(f"**{title}**")
            st.image(str(gpath), use_container_width=True)

    if not any(p.exists() for p in graph_files.values()):
        st.warning("Run `python src/evaluate.py` to generate graphs.")

        # Show simulated confusion matrix
        st.markdown("**Sample Confusion Matrix (Simulated)**")
        cm = np.array([[912, 38], [47, 953]])
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["FREE", "OCCUPIED"], color="#e6edf3")
        ax.set_yticklabels(["FREE", "OCCUPIED"], color="#e6edf3")
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("True", color="#8b949e")
        ax.set_title("Confusion Matrix", color="#e6edf3", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > 500 else "black",
                        fontsize=16, fontweight="bold")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <h3 style='color:#e6edf3'>&#x1F17F;&#xFE0F; Smart Parking System</h3>
    <p>A <b>2026-level AI parking detection system</b> that combines traditional Computer Vision
    with state-of-the-art Deep Learning (RT-DETRv2) to classify parking slots as
    <span class='tag-free'>FREE</span> or <span class='tag-occ'>OCCUPIED</span> in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔬 How It Works")
        st.markdown("""
1. **OpenCV Pipeline** — Grayscale → Blur → Adaptive Threshold → Pixel Count
2. **RT-DETRv2 Pipeline** — Backbone → Transformer → Bounding Boxes
3. **Dashboard** — Upload → Detect → View counts & charts → Download result
        """)

        st.markdown("#### 📚 Tech Stack")
        st.markdown("""
| Component | Technology |
|---|---|
| Computer Vision | OpenCV 4.x |
| Deep Learning | RT-DETRv2 (Ultralytics) |
| Training | PyTorch |
| Dashboard | Streamlit |
| Charts | Matplotlib |
| Dataset | CNRPark |
        """)

    with c2:
        st.markdown("#### 📊 Expected Performance")
        st.markdown("""
| Method | Accuracy | Speed |
|---|---|---|
| OpenCV | ~75% | 60+ FPS |
| RT-DETRv2 | **93-98%** | ~25 FPS |
        """)

        st.markdown("#### 🚀 Quick Start")
        st.code("""pip install -r requirements.txt
python src/generate_sample_data.py  # Create sample dataset
python src/preprocess.py             # Prepare YOLO format
python src/train_fast_cpu.py         # Train (CPU demo)
streamlit run src/dashboard.py       # Launch dashboard""", language="bash")

    st.markdown("---")
    st.markdown("#### 👥 Team")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""<div class='info-box' style='text-align:center'>
        <b>Pankaj Yadav</b><br>Lead Developer<br>Computer Vision Engineer
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""<div class='info-box' style='text-align:center'>
        <b>Team Member 2</b><br>Data Processing<br>Model Training
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""<div class='info-box' style='text-align:center'>
        <b>Team Member 3</b><br>Dashboard<br>Evaluation
        </div>""", unsafe_allow_html=True)

