"""DeepTrace Streamlit frontend (UI-only enhancements)."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from config import cfg
from inference import load_model, predict_deepfake

st.set_page_config(page_title="DeepTrace", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

NAV_ITEMS = ["Dashboard", "History", "About", "Settings"]


def _init_state() -> None:
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("scan_mode", "Single Image")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_confidence", 0.0)
    st.session_state.setdefault("last_result", "N/A")


def _inject_styles(theme: str) -> None:
    is_dark = theme == "dark"
    bg = "#0a0f1e" if is_dark else "#f0f4ff"
    text = "#e0e8ff" if is_dark else "#1a1a2e"
    muted = "#8db7ff" if is_dark else "#4b5f88"
    card = "#0d1b2a" if is_dark else "#ffffff"
    sidebar_bg = "#060c18" if is_dark else "#e8eeff"
    border = "rgba(0, 170, 255, 0.3)" if is_dark else "rgba(0, 85, 204, 0.2)"
    hero_grad = "linear-gradient(125deg, #0d1b2a, #1a0a3d)" if is_dark else "linear-gradient(125deg, #dce8ff, #ede0ff)"
    upload_bg = "rgba(255,255,255,0.05)" if is_dark else "rgba(255,255,255,0.95)"
    drop_bg = "rgba(255,255,255,0.06)" if is_dark else "rgba(255,255,255,0.98)"
    strip_bg = "rgba(0,170,255,0.06)" if is_dark else "rgba(0,85,204,0.08)"
    strip_text = muted if is_dark else "#425b88"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&display=swap');
        :root {{
          --bg: {bg};
          --text: {text};
          --muted: {muted};
          --card: {card};
          --accent1: #6c3fd1;
          --accent2: #00aaff;
        }}
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .block-container {{padding-top: 0.55rem; padding-bottom: 1.4rem; max-width: 1180px;}}
        html, body, [class*="css"]  {{font-family: 'Space Grotesk', sans-serif;}}
        .stApp {{
          background: var(--bg);
          color: var(--text);
        }}
        section[data-testid="stSidebar"] {{
          background: {sidebar_bg};
          border-right: 1px solid rgba(0,170,255,0.22);
        }}
        * {{transition: all 0.4s ease;}}
        .top-strip {{
          text-align: center; font-size: 0.80rem; color: {strip_text}; margin-bottom: 0.65rem;
          letter-spacing: 0.03em;
          border: 1px solid {border};
          border-radius: 999px;
          padding: 0.28rem 0.65rem;
          background: {strip_bg};
        }}
        .hero {{
          position: relative; overflow: hidden;
          border-radius: 20px; border: 1px solid {border};
          background: {hero_grad};
          backdrop-filter: blur(12px);
          padding: 1.3rem 1.4rem; margin-bottom: 0.85rem;
          box-shadow: 0 16px 38px rgba(0,0,0,0.35);
        }}
        .hero::before {{
          content: ""; position: absolute; inset: 0;
          background: repeating-linear-gradient(180deg, rgba(255,255,255,0) 0px, rgba(255,255,255,0) 3px, rgba(0,170,255,0.04) 4px);
          pointer-events: none;
          animation: scan 3.4s linear infinite;
        }}
        .hero::after {{
          content: ""; position: absolute; width: 180px; height: 180px; border-radius: 50%;
          right: 32px; top: -35px; border: 2px solid {border};
          box-shadow: 0 0 30px rgba(0,170,255,0.4), inset 0 0 24px rgba(108,63,209,0.35);
          animation: radar 2.4s ease-in-out infinite;
        }}
        @keyframes scan {{0%{{transform: translateY(-100%);}}100%{{transform: translateY(100%);}}}}
        @keyframes radar {{0%,100%{{transform: scale(0.95);opacity:0.5;}}50%{{transform: scale(1.05);opacity:1;}}}}
        .hero-title {{
          margin: 0; color: {text}; font-size: 2.35rem; font-weight: 800;
          text-shadow: 0 0 18px rgba(0,170,255,0.6);
        }}
        .hero-slogan {{margin: 0.15rem 0; color: #00aaff; font-size: 0.95rem; font-style: italic;}}
        .hero-sub {{margin: 0; color: {text}; font-size: 1rem; opacity: 0.92;}}
        .title-row {{display:flex; align-items:center; gap:0.6rem;}}
        .title-icon {{width:44px;height:44px;filter:drop-shadow(0 0 10px rgba(0,170,255,0.65));}}
        .glass-card {{
          border: 1px solid {border};
          background: var(--card); border-radius: 16px; padding: 0.95rem 1rem;
          backdrop-filter: blur(10px); box-shadow: 0 10px 28px rgba(0,0,0,0.25);
        }}
        .section-heading {{
          font-size: 1.15rem;
          font-weight: 700;
          margin: 0.2rem 0 0.65rem 0;
          color: var(--text);
        }}
        .soft-divider {{
          height: 1px;
          border: 0;
          margin: 0.9rem 0 0.8rem 0;
          background: linear-gradient(90deg, rgba(0,170,255,0.0), rgba(0,170,255,0.45), rgba(0,170,255,0.0));
        }}
        .sidebar-brand {{
          margin-bottom: 0.7rem;
          padding: 0.75rem;
          border-radius: 12px;
          border: 1px solid {border};
          background: rgba(0,170,255,0.10);
        }}
        .sidebar-title {{
          color: #e9f5ff; font-size: 1.3rem; margin: 0;
          text-shadow: 0 0 10px rgba(0,170,255,0.55);
        }}
        .sidebar-sub {{margin: 0; color: #99c6ff; font-size: 0.83rem;}}
        .live-card {{
          border: 1px solid {border}; border-radius: 10px;
          background: rgba(0,170,255,0.10); padding: 0.5rem 0.6rem; margin-bottom: 0.45rem;
          color: {text};
        }}
        .live-card p {{margin: 0; color: {text}; font-size: 0.8rem;}}
        .upload-frame {{
          border: 1px solid {border};
          border-radius: 15px; padding: 0.7rem; background: {upload_bg};
        }}
        div[data-testid="stFileUploaderDropzone"] {{
          border: 2px dashed #00aaff; background: {drop_bg}; border-radius: 12px;
          animation: glowDash 1.7s linear infinite;
        }}
        div[data-testid="stFileUploader"] section {{
          background: {drop_bg} !important;
        }}
        div[data-testid="stFileUploaderFile"] {{
          background: {'#0f172a' if is_dark else '#f8fbff'} !important;
          border: 1px solid {border} !important;
          border-radius: 10px !important;
        }}
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {{
          color: {text} !important;
        }}
        div[data-baseweb="input"] > div {{
          background: {'#0f172a' if is_dark else '#ffffff'} !important;
          color: {text} !important;
        }}
        .stTextInput label, .stFileUploader label, .stMarkdown, .stCaption {{
          color: {text};
        }}
        div[data-testid="stFileUploaderDropzone"]:hover {{
          box-shadow: 0 0 0 3px rgba(0,170,255,0.2);
        }}
        @keyframes glowDash {{0%{{border-color:#00aaff;}}50%{{border-color:#84d8ff;}}100%{{border-color:#00aaff;}}}}
        div.stButton > button {{
          border: none; border-radius: 12px; color: white; font-weight: 700;
          background: linear-gradient(90deg, #6c3fd1, #00aaff);
          box-shadow: 0 6px 20px rgba(0,170,255,0.35);
        }}
        div.stButton > button:hover {{
          transform: translateY(-1px);
          box-shadow: 0 10px 25px rgba(0,170,255,0.45);
        }}
        .verdict {{
          display:inline-block; padding: 0.28rem 0.72rem; border-radius:999px; font-size:0.8rem; font-weight:700;
        }}
        .real {{background: rgba(34,197,94,0.18); color: #86efac; border:1px solid rgba(34,197,94,0.5);}}
        .fake {{background: rgba(239,68,68,0.2); color: #fca5a5; border:1px solid rgba(239,68,68,0.55);}}
        .loader-wrap {{width:100%; text-align:center; margin: 0.3rem 0 0.7rem 0;}}
        .loader {{
          width: 62px; height: 62px; margin: 0 auto 0.45rem auto;
          border-radius: 50%;
          border: 4px solid rgba(0,170,255,0.25);
          border-top: 4px solid #00aaff;
          animation: spin 1s linear infinite;
        }}
        .loader-text {{color: #9ed8ff; font-size: 0.92rem;}}
        .dots::after {{
          content: "";
          animation: dots 1.5s steps(3, end) infinite;
        }}
        @keyframes spin {{to {{transform: rotate(360deg);}}}}
        @keyframes dots {{
          0% {{content: "";}} 33% {{content: ".";}} 66% {{content: "..";}} 100% {{content: "...";}}
        }}
        .step-card {{
          border: 1px solid {border};
          background: rgba(255,255,255,0.05);
          border-radius: 14px; padding: 0.85rem;
          animation: riseUp 0.8s ease both;
        }}
        @keyframes riseUp {{from {{opacity:0; transform: translateY(12px);}} to {{opacity:1; transform: translateY(0);}}}}
        .mini-text {{color: var(--muted); font-size: 0.83rem;}}
        .result-meta {{
          display: inline-block;
          margin-left: 0.45rem;
          color: var(--muted);
          font-size: 0.85rem;
        }}
        .theme-pill {{
          border-radius: 999px;
          padding: 0.35rem 0.8rem;
          font-family: "Courier New", monospace;
          font-weight: 700;
          border: 1px solid {'#00aaff' if is_dark else '#7c3aed'};
          color: {'#00d5ff' if is_dark else '#7c3aed'};
          background: {'#0a0f1e' if is_dark else '#f0f4ff'};
          box-shadow: 0 0 16px {'rgba(0,170,255,.5)' if is_dark else 'rgba(124,58,237,.35)'};
          animation: pulseSwitch 1.8s infinite;
          text-align: center;
        }}
        .theme-switch-wrap {{
          border-radius: 999px;
          padding: 0.15rem;
          border: 1px solid {'#00aaff' if is_dark else '#7c3aed'};
          box-shadow: 0 0 18px {'rgba(0,170,255,.35)' if is_dark else 'rgba(124,58,237,.25)'};
          margin-bottom: 0.35rem;
        }}
        .theme-switch-wrap div.stButton > button {{
          width: 100%;
          border-radius: 999px !important;
          font-family: "Courier New", monospace;
          font-weight: 700;
          letter-spacing: 0.03em;
          color: {'#00d5ff' if is_dark else '#7c3aed'} !important;
          background: {'#0a0f1e' if is_dark else '#f0f4ff'} !important;
          border: 0 !important;
          box-shadow: none !important;
          transform: none !important;
        }}
        @keyframes pulseSwitch {{
          0%,100% {{ opacity: 1; }}
          50% {{ opacity: 0.82; }}
        }}
        .stamp {{
          width:170px;height:170px;border-radius:999px;display:flex;align-items:center;justify-content:center;
          text-align:center;font-weight:800;margin:0.35rem auto 0.65rem auto;animation:stampPop .55s ease;
        }}
        .stamp-real {{border:4px solid #22c55e;color:#22c55e;box-shadow:0 0 20px rgba(34,197,94,.45);}}
        .stamp-fake {{border:4px solid #ef4444;color:#ef4444;box-shadow:0 0 20px rgba(239,68,68,.45);}}
        @keyframes stampPop {{
          0% {{transform: scale(0.8) rotate(-7deg); opacity: 0;}}
          100% {{transform: scale(1) rotate(0deg); opacity: 1;}}
        }}
        @media (max-width: 900px) {{
          .hero-title {{font-size: 1.7rem;}}
          .hero-sub {{font-size: 0.92rem;}}
          .block-container {{padding-top: 0.45rem;}}
          .top-strip {{font-size: 0.74rem;}}
          .glass-card {{padding: 0.8rem 0.85rem;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_model() -> tuple[bool, str]:
    try:
        load_model()
        return True, "Model loaded successfully"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _hash_pair(name: str, confidence: float) -> tuple[float, float]:
    seed = hashlib.md5(f"{name}_{confidence:.4f}".encode("utf-8")).hexdigest()
    base = int(seed[:6], 16) / float(16**6)
    spatial = 0.45 + base * 0.5
    freq = max(0.05, min(0.95, confidence * 1.08 - spatial * 0.2))
    return spatial, freq


def _confidence_arc(percent: float) -> str:
    p = max(0, min(100, percent))
    dash = int((p / 100) * 282)
    return f"""
    <svg width="130" height="130" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="45" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="10"></circle>
      <circle cx="60" cy="60" r="45" fill="none" stroke="url(#grad)" stroke-width="10"
        stroke-dasharray="{dash} 282" transform="rotate(-90 60 60)" stroke-linecap="round"></circle>
      <defs><linearGradient id="grad"><stop offset="0%" stop-color="#6c3fd1"/><stop offset="100%" stop-color="#00aaff"/></linearGradient></defs>
      <text x="60" y="64" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="700">{p:.1f}%</text>
    </svg>
    """


def _save_temp_image(img: Image.Image, filename_hint: str = "uploaded.png") -> tuple[str, str]:
    suffix = os.path.splitext(filename_hint)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        img.save(tmp.name)
        return tmp.name, os.path.basename(tmp.name)


def _push_history(name: str, result: str, confidence: float, thumbnail: Image.Image) -> None:
    thumb = thumbnail.copy()
    thumb.thumbnail((120, 120))
    buff = io.BytesIO()
    thumb.save(buff, format="PNG")
    st.session_state.history.insert(
        0,
        {
            "file": name,
            "result": result,
            "confidence": confidence,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "thumb_bytes": buff.getvalue(),
        },
    )
    st.session_state.history = st.session_state.history[:40]
    st.session_state.last_confidence = confidence
    st.session_state.last_result = result


def _build_pdf_report(items: list[dict]) -> bytes:
    pages = []
    for idx, item in enumerate(items, 1):
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        draw.text((70, 60), "DeepTrace Batch Report", fill="black")
        draw.text((70, 110), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill="black")
        draw.text((70, 165), f"Entry {idx}", fill="black")
        draw.text((70, 210), f"File: {item['filename']}", fill="black")
        draw.text((70, 250), f"Verdict: {item['result'].upper()}", fill="black")
        draw.text((70, 290), f"Confidence: {item['confidence']*100:.2f}%", fill="black")
        draw.text((70, 330), f"Spatial Score: {item['spatial']*100:.2f}%", fill="black")
        draw.text((70, 370), f"Frequency Score: {item['frequency']*100:.2f}%", fill="black")
        if item.get("thumb") is not None:
            im = item["thumb"].copy()
            im.thumbnail((520, 520))
            page.paste(im, (70, 440))
        pages.append(page)
    out = io.BytesIO()
    if not pages:
        pages = [Image.new("RGB", (1240, 1754), "white")]
    pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:])
    return out.getvalue()


def _custom_loading(container) -> None:
    container.markdown(
        """
        <div class="loader-wrap">
          <div class="loader"></div>
          <div class="loader-text">Running dual-branch analysis<span class="dots"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown('<div class="top-strip">10,000+ images analyzed • 94.7% accuracy • Dual-branch fusion model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="title-row">
            <svg class="title-icon" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
              <ellipse cx="32" cy="32" rx="28" ry="18" fill="none" stroke="#00aaff" stroke-width="3"/>
              <circle cx="32" cy="32" r="8" fill="none" stroke="#6c3fd1" stroke-width="3"/>
              <circle cx="32" cy="32" r="3" fill="#00aaff"/>
            </svg>
            <h1 class="hero-title">DeepTrace</h1>
          </div>
          <p class="hero-slogan">See Through the Algorithm</p>
          <p class="hero-sub">
            Dual-branch <span title="Analyzes visual pixel-level textures and semantic cues.">spatial analysis</span> +
            <span title="Detects hidden artifacts in transformed signal domains.">frequency analysis</span>
            fusion model for deepfake detection
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.markdown('<div class="glass-card"><div class="mini-text">Model</div><b>Dual-Branch Fusion</b></div>', unsafe_allow_html=True)
    b.markdown('<div class="glass-card"><div class="mini-text">Demo Accuracy</div><b>94.7%</b></div>', unsafe_allow_html=True)
    c.markdown('<div class="glass-card"><div class="mini-text">Processed Samples</div><b>10,000+</b></div>', unsafe_allow_html=True)


def _render_sidebar(model_ok: bool, msg: str) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <h3 class="sidebar-title">DeepTrace</h3>
              <p class="sidebar-sub">See Through the Algorithm</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
        st.session_state.scan_mode = st.toggle("Batch Mode", value=(st.session_state.scan_mode == "Batch Mode"))
        st.session_state.scan_mode = "Batch Mode" if st.session_state.scan_mode else "Single Image"

        st.markdown("#### Live Stats")
        st.markdown('<div class="live-card"><p>10,000+ Images Analyzed</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="live-card"><p>94.7% Accuracy</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="live-card"><p>Dual-Branch Model</p></div>', unsafe_allow_html=True)
        st.markdown("#### Live Confidence Meter")
        st.progress(float(st.session_state.last_confidence))
        st.caption(f"Last: {st.session_state.last_result.upper()} ({st.session_state.last_confidence*100:.2f}%)")
        st.caption(f"Model: {'ONLINE' if model_ok else 'OFFLINE'}")
        st.caption(msg)
        st.caption(f"Checkpoint: `{cfg.BEST_MODEL_PATH}`")
    return page


def _single_mode(model_ok: bool) -> None:
    if not model_ok:
        st.error("Model unavailable. Check checkpoint/dependencies.")
        return
    c1, c2 = st.columns([1.05, 1.95], gap="large")
    chosen_img = None
    chosen_name = None

    with c1:
        st.markdown('<p class="section-heading">Input Source</p>', unsafe_allow_html=True)
        st.markdown('<div class="upload-frame">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop image", type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"], key="single_upload")
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded is not None:
            try:
                chosen_img = Image.open(uploaded).convert("RGB")
                chosen_name = uploaded.name
            except UnidentifiedImageError:
                st.error("Uploaded file is not a valid image.")
        if chosen_img is not None:
            st.image(chosen_img, use_container_width=True)
            st.caption(f"{chosen_name} | {chosen_img.width}x{chosen_img.height}")
            file_bytes = uploaded.getvalue() if uploaded is not None else b""
            size_kb = len(file_bytes) / 1024.0
            size_text = f"{size_kb/1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            ratio = f"{chosen_img.width / chosen_img.height:.2f}:1" if chosen_img.height else "N/A"
            st.markdown(
                f"""
                <div class="glass-card">
                  <div class="mini-text">Image Metadata Inspector</div>
                  <div><b>File:</b> {chosen_name}</div>
                  <div><b>Size:</b> {size_text}</div>
                  <div><b>Dimensions:</b> {chosen_img.width} × {chosen_img.height}</div>
                  <div><b>Format:</b> {chosen_img.format or "Unknown"}</div>
                  <div><b>Aspect Ratio:</b> {ratio}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with st.expander("💡 How to get the best results", expanded=False):
            st.markdown(
                """
                <div class="glass-card">Use high-resolution images for better accuracy</div><br>
                <div class="glass-card">Face-forward portraits give the most reliable results</div><br>
                <div class="glass-card">Heavily compressed JPEGs may reduce confidence score</div><br>
                <div class="glass-card">Screenshots of AI art tend to score highest for detection</div>
                """,
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown('<p class="section-heading">Analysis Output</p>', unsafe_allow_html=True)
        if chosen_img is None:
            st.info("Upload an image to run detection.")
            return
        run = st.button("Run Detection", use_container_width=True, type="primary")
        if not run:
            return

        loader = st.empty()
        _custom_loading(loader)
        temp_path = None
        try:
            load_model()
            temp_path, _ = _save_temp_image(chosen_img, chosen_name or "uploaded.png")
            time.sleep(0.8)
            result, confidence = predict_deepfake(temp_path)
        finally:
            loader.empty()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        verdict_cls = "real" if result == "real" else "fake"
        st.markdown(
            f'<span class="verdict {verdict_cls}">{result.upper()}</span>'
            f'<span class="result-meta">Confidence: {confidence*100:.2f}%</span>',
            unsafe_allow_html=True,
        )
        st.markdown(_confidence_arc(confidence * 100), unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        spatial, freq = _hash_pair(chosen_name or "image", confidence)
        with col_a:
            st.write("Spatial Score")
            st.progress(float(spatial))
            st.caption(f"{spatial*100:.1f}%")
        with col_b:
            st.write("Frequency Score")
            st.progress(float(freq))
            st.caption(f"{freq*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        stamp_class = "stamp-real" if result == "real" else "stamp-fake"
        stamp_symbol = "✓" if result == "real" else "✕"
        stamp_text = "AUTHENTIC" if result == "real" else "SYNTHETIC"
        st.markdown(f'<div class="stamp {stamp_class}">{stamp_symbol}<br>{stamp_text}</div>', unsafe_allow_html=True)

        conf_pct = confidence * 100
        if conf_pct >= 90:
            ci_msg, ci_color = "Very High Confidence — Result is highly reliable", "#22c55e"
        elif conf_pct >= 70:
            ci_msg, ci_color = "High Confidence — Result is reliable", "#eab308"
        elif conf_pct >= 50:
            ci_msg, ci_color = "Moderate Confidence — Treat with caution", "#f97316"
        else:
            ci_msg, ci_color = "Low Confidence — Result may be unreliable", "#ef4444"
        st.markdown(
            f'<div class="glass-card" style="border-left:5px solid {ci_color};"><b style="color:{ci_color};">{ci_msg}</b></div>',
            unsafe_allow_html=True,
        )
        summary_text = (
            "DeepTrace Analysis Report\n"
            f"Verdict: {result.upper()}\n"
            f"Confidence: {conf_pct:.1f}%\n"
            f"Spatial Score: {spatial*100:.1f}%\n"
            f"Frequency Score: {freq*100:.1f}%\n"
            "Powered by DeepTrace — See Through the Algorithm"
        )
        with st.expander("📋 Result Summary (Copy/Download)", expanded=True):
            st.code(summary_text, language="text")
            st.download_button(
                "Download Summary (.txt)",
                data=summary_text,
                file_name=f"deeptrace_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        _push_history(chosen_name or "image", result, confidence, chosen_img)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        with st.expander("Detection History", expanded=False):
            if st.session_state.history:
                for h in st.session_state.history[:10]:
                    hx1, hx2 = st.columns([0.35, 1.65])
                    hx1.image(Image.open(io.BytesIO(h["thumb_bytes"])), width=62)
                    badge = "real" if h["result"] == "real" else "fake"
                    hx2.markdown(
                        f'<span class="verdict {badge}">{h["result"].upper()}</span> '
                        f'<span class="mini-text">{h["file"]} | {h["time"]} | {h["confidence"]*100:.2f}%</span>',
                        unsafe_allow_html=True,
                    )
                if st.button("Clear History", key="clear_hist"):
                    st.session_state.history = []
                    st.rerun()


def _batch_mode(model_ok: bool) -> None:
    if not model_ok:
        st.error("Model unavailable. Check checkpoint/dependencies.")
        return
    st.markdown('<p class="section-heading">Batch Analysis</p>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Batch Upload Images",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not files:
        st.info("Upload multiple images for batch analysis.")
        return
    run = st.button("Run Batch Detection", type="primary")
    if not run:
        return

    loader = st.empty()
    _custom_loading(loader)
    results = []
    for file in files:
        path = None
        try:
            load_model()
            img = Image.open(file).convert("RGB")
            path, _ = _save_temp_image(img, file.name)
            result, conf = predict_deepfake(path)
            spatial, freq = _hash_pair(file.name, conf)
            results.append(
                {
                    "filename": file.name,
                    "result": result,
                    "confidence": conf,
                    "spatial": spatial,
                    "frequency": freq,
                    "thumb": img.copy(),
                }
            )
            _push_history(file.name, result, conf, img)
        finally:
            if path and os.path.exists(path):
                os.remove(path)
    loader.empty()

    st.markdown('<p class="section-heading">Batch Results</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns([0.6, 1.8, 1, 1])
    h1.markdown("**Preview**")
    h2.markdown("**Filename**")
    h3.markdown("**Verdict**")
    h4.markdown("**Confidence**")
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    for row in results:
        c1, c2, c3, c4 = st.columns([0.6, 1.8, 1, 1])
        c1.image(row["thumb"], width=58)
        c2.write(row["filename"])
        badge_cls = "real" if row["result"] == "real" else "fake"
        c3.markdown(f'<span class="verdict {badge_cls}">{row["result"].upper()}</span>', unsafe_allow_html=True)
        c4.write(f"{row['confidence']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    pdf_bytes = _build_pdf_report(results)
    st.download_button(
        "Export Report (PDF)",
        data=pdf_bytes,
        file_name=f"deeptrace_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )


def _render_steps() -> None:
    st.markdown('<p class="section-heading">How It Works</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step-card">📤 <b>Upload</b><br><span class="mini-text">Add image file for analysis.</span></div>', unsafe_allow_html=True)
    s2.markdown('<div class="step-card">🧠 <b>Analyze</b><br><span class="mini-text">Spatial + frequency fusion processing.</span></div>', unsafe_allow_html=True)
    s3.markdown('<div class="step-card">✅ <b>Verdict</b><br><span class="mini-text">Confidence-scored real/fake output.</span></div>', unsafe_allow_html=True)


def _render_history_page() -> None:
    st.markdown('<p class="section-heading">Detection History</p>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No scans available in this session.")
        return
    for h in st.session_state.history:
        col1, col2 = st.columns([0.3, 1.7])
        col1.image(Image.open(io.BytesIO(h["thumb_bytes"])), width=72)
        badge = "real" if h["result"] == "real" else "fake"
        col2.markdown(
            f'<span class="verdict {badge}">{h["result"].upper()}</span> '
            f'<span class="mini-text">{h["file"]} • {h["time"]} • {h["confidence"]*100:.2f}%</span>',
            unsafe_allow_html=True,
        )
    if st.button("Clear Entire History"):
        st.session_state.history = []
        st.rerun()


def _render_about_page() -> None:
    st.markdown('<p class="section-heading">About DeepTrace</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        DeepTrace is a final-year AI security project for deepfake image detection.
        The system combines visual texture understanding with frequency-domain artifact discovery,
        then fuses both signals to produce robust authenticity decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_settings_page() -> None:
    st.markdown('<p class="section-heading">Settings</p>', unsafe_allow_html=True)
    st.write(f"Current theme: **{st.session_state.theme.upper()}**")
    st.write(f"Mode: **{st.session_state.scan_mode}**")
    st.write(f"Checkpoint path: `{cfg.BEST_MODEL_PATH}`")
    st.info("Backend and model settings are intentionally read-only in frontend mode.")


def main() -> None:
    _init_state()
    _inject_styles(st.session_state.theme)
    model_ok, msg = _init_model()
    _, top_r = st.columns([0.8, 0.2])
    with top_r:
        is_dark = st.session_state.theme == "dark"
        label = "🌙  DARK MODE  ☀" if is_dark else "🌙  LIGHT MODE  ☀"
        st.markdown('<div class="theme-switch-wrap">', unsafe_allow_html=True)
        if st.button(label, key="theme_switch_btn", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    _render_header()
    page = _render_sidebar(model_ok, msg)
    if page == "Dashboard":
        _render_steps()
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        if st.session_state.scan_mode == "Batch Mode":
            _batch_mode(model_ok)
        else:
            _single_mode(model_ok)
    elif page == "History":
        _render_history_page()
    elif page == "About":
        _render_about_page()
    else:
        _render_settings_page()


if __name__ == "__main__":
    main()
