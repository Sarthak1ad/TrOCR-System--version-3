"""
finetune/prepare_dataset.py

Turns full-page photos of your handwriting into individual line-crop images,
ready for labeling. Reuses the same PaddleOCR detection already in the main
pipeline — it finds each line, we just save each one as its own file instead
of feeding it to TrOCR.

USAGE:
  1. Drop full-page photos of your handwriting into finetune/data/raw_photos/
  2. Run: python finetune/prepare_dataset.py
  3. Output:
     - finetune/data/line_crops/*.png   (one file per detected line)
     - finetune/data/labels.csv          (template: filename, text)
  4. Open labels.csv and fill in the "text" column with the correct
     transcription for each line crop (open the matching .png to read it).
"""

import os
import sys
import csv
import glob

import cv2

# Reuse the detection + cropping logic already built for the main pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_pipeline import get_detector, _extract_boxes, _crop_box  # noqa: E402
from preprocessing import preprocess_image  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PHOTOS_DIR = os.path.join(THIS_DIR, "data", "raw_photos")
LINE_CROPS_DIR = os.path.join(THIS_DIR, "data", "line_crops")
LABELS_CSV = os.path.join(THIS_DIR, "data", "labels.csv")


def prepare_dataset(apply_preprocessing: bool = True, min_line_height: int = 15):
    os.makedirs(LINE_CROPS_DIR, exist_ok=True)

    photo_paths = sorted(
        glob.glob(os.path.join(RAW_PHOTOS_DIR, "*.jpg"))
        + glob.glob(os.path.join(RAW_PHOTOS_DIR, "*.jpeg"))
        + glob.glob(os.path.join(RAW_PHOTOS_DIR, "*.png"))
    )

    if not photo_paths:
        print(f"No photos found in {RAW_PHOTOS_DIR}")
        print("Add some full-page photos of your handwriting there first.")
        return

    detector = get_detector(lang="en")
    rows = []
    total_lines = 0

    for photo_path in photo_paths:
        stem = os.path.splitext(os.path.basename(photo_path))[0]
        print(f"Processing {photo_path} ...")

        if apply_preprocessing:
            image = preprocess_image(photo_path)
        else:
            image = cv2.imread(photo_path)
            if image is None:
                print(f"  Could not read {photo_path}, skipping.")
                continue

        tmp_path = os.path.join(THIS_DIR, "_tmp_prep.jpg")
        cv2.imwrite(tmp_path, image)
        raw_result = detector.ocr(tmp_path, cls=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        boxes = _extract_boxes(raw_result)
        if not boxes:
            print(f"  No text lines detected in {photo_path}.")
            continue

        line_idx = 0
        for box in boxes:
            crop = _crop_box(image, box)
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < min_line_height:
                continue  # skip slivers/noise, not real text lines

            filename = f"{stem}_{line_idx:03d}.png"
            out_path = os.path.join(LINE_CROPS_DIR, filename)
            cv2.imwrite(out_path, crop)
            rows.append({"filename": filename, "text": ""})
            line_idx += 1
            total_lines += 1

        print(f"  -> {line_idx} line(s) extracted")

    # Don't overwrite existing labels if the user already started labeling —
    # append only new rows instead.
    existing_filenames = set()
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_filenames.add(row["filename"])

    new_rows = [r for r in rows if r["filename"] not in existing_filenames]

    write_header = not os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print(f"\nDone. {total_lines} line(s) extracted this run, {len(new_rows)} new row(s) added to labels.csv.")
    print(f"Line crops: {LINE_CROPS_DIR}")
    print(f"Labels file: {LABELS_CSV}")
    print("\nNext step: open labels.csv, look at each line_crops/*.png, and type the correct text in the 'text' column.")


if __name__ == "__main__":
    prepare_dataset()
