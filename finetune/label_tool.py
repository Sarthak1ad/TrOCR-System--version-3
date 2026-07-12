"""
finetune/label_tool.py

A fast labeling UI for your line crops: shows one image at a time, you type
the correct text, hit Save & Next. Much quicker than manually matching
filenames in Excel to images in a file browser.

USAGE:
  streamlit run finetune/label_tool.py
"""

import os
import pandas as pd
import streamlit as st
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_CROPS_DIR = os.path.join(THIS_DIR, "data", "line_crops")
LABELS_CSV = os.path.join(THIS_DIR, "data", "labels.csv")

st.set_page_config(page_title="Handwriting Labeler", page_icon="✍️", layout="centered")


def load_labels():
    if not os.path.exists(LABELS_CSV):
        st.error(f"No labels.csv found at {LABELS_CSV}. Run prepare_dataset.py first.")
        st.stop()
    df = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
    return df


def save_labels(df):
    df.to_csv(LABELS_CSV, index=False)


if "df" not in st.session_state:
    st.session_state.df = load_labels()
if "idx" not in st.session_state:
    # jump straight to the first unlabeled row on first load
    df = st.session_state.df
    unlabeled = df.index[df["text"].str.strip() == ""]
    st.session_state.idx = int(unlabeled[0]) if len(unlabeled) > 0 else 0

df = st.session_state.df
total = len(df)
labeled_count = int((df["text"].str.strip() != "").sum())

st.title("✍️ Handwriting Labeler")
st.progress(labeled_count / total if total else 0, text=f"{labeled_count} / {total} labeled")

idx = st.session_state.idx
idx = max(0, min(idx, total - 1))
st.session_state.idx = idx

row = df.iloc[idx]
image_path = os.path.join(LINE_CROPS_DIR, row["filename"])

col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("⬅ Previous", use_container_width=True, disabled=(idx == 0)):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()
with col_nav2:
    st.markdown(f"<div style='text-align:center'>Line {idx + 1} of {total} — {row['filename']}</div>", unsafe_allow_html=True)
with col_nav3:
    if st.button("Next ➡", use_container_width=True, disabled=(idx == total - 1)):
        st.session_state.idx = min(total - 1, idx + 1)
        st.rerun()

st.divider()

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
else:
    st.warning(f"Image file missing: {image_path}")

text_value = st.text_input(
    "Type the correct text for this line:",
    value=row["text"],
    key=f"text_input_{idx}",
)

col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
with col_save1:
    if st.button("💾 Save & Next", type="primary", use_container_width=True):
        st.session_state.df.at[idx, "text"] = text_value
        save_labels(st.session_state.df)
        if idx < total - 1:
            st.session_state.idx = idx + 1
        st.rerun()
with col_save2:
    if st.button("Save (stay here)", use_container_width=True):
        st.session_state.df.at[idx, "text"] = text_value
        save_labels(st.session_state.df)
        st.success("Saved.")
with col_save3:
    if st.button("Skip (leave blank)", use_container_width=True):
        if idx < total - 1:
            st.session_state.idx = idx + 1
        st.rerun()

st.divider()
st.caption(
    "Tip: press Tab then Enter to move focus off the text box before clicking "
    "Save & Next if the button doesn't seem to respond to a plain Enter key."
)

with st.expander("Jump to a specific line number"):
    jump_to = st.number_input("Line number", min_value=1, max_value=total, value=idx + 1)
    if st.button("Jump"):
        st.session_state.idx = jump_to - 1
        st.rerun()

with st.expander("Jump to next unlabeled line"):
    if st.button("Find next unlabeled"):
        unlabeled = df.index[df["text"].str.strip() == ""]
        remaining = unlabeled[unlabeled > idx]
        if len(remaining) > 0:
            st.session_state.idx = int(remaining[0])
        elif len(unlabeled) > 0:
            st.session_state.idx = int(unlabeled[0])
        else:
            st.success("Everything is labeled!")
        st.rerun()
