"""
web/app.py
Streamlit UI for the OCR pipeline: drag-and-drop upload, preview, run OCR,
show extracted text + per-line confidence, export as JSON/CSV/TXT.

Run with:
  streamlit run web/app.py
"""

import os
import sys
import json
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_pipeline import run_ocr  # noqa: E402

st.set_page_config(page_title="OCR Text Extractor", page_icon="📝", layout="wide")

st.title("📝 Document / Handwriting OCR")
st.caption("Upload a photo or scan — get back clean, extracted text with confidence scores.")

with st.sidebar:
    st.header("Settings")
    min_confidence = st.slider("Minimum confidence to keep a line", 0.0, 1.0, 0.70, 0.05)
    apply_preprocessing = st.checkbox("Apply preprocessing (denoise, deskew, contrast)", value=True)
    lang = st.selectbox("Language", ["en", "ch", "fr", "german", "korean", "japan"], index=0)

uploaded_files = st.file_uploader(
    "Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.subheader("Uploaded image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with col2:
            st.subheader("Extracted text")
            with st.spinner("Reading text..."):
                try:
                    result = run_ocr(
                        tmp_path,
                        apply_preprocessing=apply_preprocessing,
                        min_confidence=min_confidence,
                        lang=lang,
                    )
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    os.remove(tmp_path)
                    continue

            st.text_area("Full extracted text", result["text"], height=180)
            st.caption(f"Processed in {result['processing_time_ms']} ms")

            if result["lines"]:
                st.write("**Per-line confidence**")
                for line in result["lines"]:
                    flag = "⚠️ " if line["confidence"] < min_confidence else ""
                    st.progress(line["confidence"], text=f"{flag}{line['text']}  ({line['confidence']:.0%})")
            else:
                st.info("No text detected.")

            if result["low_confidence_lines"]:
                st.warning(f"{len(result['low_confidence_lines'])} line(s) below confidence threshold — review manually.")

            # Export options
            export_col1, export_col2, export_col3 = st.columns(3)
            with export_col1:
                st.download_button(
                    "⬇ Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.json",
                    mime="application/json",
                )
            with export_col2:
                if result["lines"]:
                    df = pd.DataFrame(result["lines"])[["text", "confidence"]]
                    st.download_button(
                        "⬇ Download CSV",
                        data=df.to_csv(index=False),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.csv",
                        mime="text/csv",
                    )
            with export_col3:
                st.download_button(
                    "⬇ Download TXT",
                    data=result["text"],
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.txt",
                    mime="text/plain",
                )

        os.remove(tmp_path)
else:
    st.info("Upload one or more images to get started.")
