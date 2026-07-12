"""
ocr_pipeline.py
End-to-end OCR pipeline for HANDWRITING: preprocessing -> PaddleOCR (line
DETECTION only) -> TrOCR (handwriting-specialized RECOGNITION) ->
confidence filtering -> structured output.

Why hybrid?
- PaddleOCR is excellent at finding WHERE text lines are (detection), even
  on messy notebook photos, uneven lighting, skewed pages, etc.
- PaddleOCR's built-in recognition model is trained mostly on printed /
  scene text (signs, labels, forms) and struggles with cursive handwriting.
- TrOCR (Microsoft) is a transformer model fine-tuned specifically on the
  IAM Handwriting Database, so it reads cursive/messy handwriting far more
  accurately than PaddleOCR's default recognizer.

So: PaddleOCR finds each line -> we crop it -> TrOCR reads that crop.
"""

import os

# PaddlePaddle and PyTorch each bundle their own copy of the OpenMP runtime
# (libiomp5md.dll on Windows). Loading both in the same process triggers
# OMP Error #15. This is the standard, safe-enough workaround — must be
# set BEFORE torch/paddleocr get imported anywhere.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from preprocessing import preprocess_image

# Both models are loaded once and reused (loading per-request is slow)
_DETECTOR = None
_TROCR_PROCESSOR = None
_TROCR_MODEL = None
_TORCH_DEVICE = None


def get_detector(lang: str = "en"):
    """Lazily loads PaddleOCR for LINE DETECTION only (rec=False at call time)."""
    global _DETECTOR
    if _DETECTOR is None:
        from paddleocr import PaddleOCR
        _DETECTOR = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, enable_mkldnn=False)
    return _DETECTOR


def get_recognizer(model_name: str = "microsoft/trocr-base-handwritten"):
    """
    Lazily loads TrOCR (processor + model) for handwriting recognition.

    Model options (swap in get_recognizer call if needed):
      - "microsoft/trocr-small-handwritten"  -> faster, smaller download, lower accuracy (default)
      - "microsoft/trocr-base-handwritten"   -> good balance, ~1.3GB download
      - "microsoft/trocr-large-handwritten"  -> best accuracy, slow, large download
    """
    global _TROCR_PROCESSOR, _TROCR_MODEL, _TORCH_DEVICE
    if _TROCR_MODEL is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(model_name)
        _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(model_name)
        _TROCR_MODEL.to(_TORCH_DEVICE)
        _TROCR_MODEL.eval()
    return _TROCR_PROCESSOR, _TROCR_MODEL


def _extract_boxes(raw_result) -> list:
    """
    Pulls box coordinates out of PaddleOCR's normal (full detect+recognize)
    result. We only need the box positions here — PaddleOCR's own
    recognized text is discarded, since TrOCR re-reads each crop for
    handwriting accuracy.

    Uses len()/indexing only, never `if some_array:`-style truthiness,
    since PaddleOCR's boxes are numpy arrays and Python can't evaluate the
    truthiness of a multi-element array (that's the "ambiguous truth value"
    error).
    """
    if raw_result is None or len(raw_result) == 0:
        return []

    detections = raw_result[0]
    if detections is None or len(detections) == 0:
        return []

    boxes = []
    for item in detections:
        box = item[0]  # item is [box, (text, confidence)]; we only want box
        boxes.append(box)
    return boxes


def _crop_box(image: np.ndarray, box: list) -> np.ndarray:
    """
    Crops a (possibly rotated) quadrilateral text region out of the image
    using a perspective warp, so skewed/angled lines still come out straight
    before being handed to TrOCR.
    """
    pts = np.array(box, dtype="float32")

    width_a = np.linalg.norm(pts[0] - pts[1])
    width_b = np.linalg.norm(pts[3] - pts[2])
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(pts[0] - pts[3])
    height_b = np.linalg.norm(pts[1] - pts[2])
    max_height = max(int(height_a), int(height_b), 1)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def _recognize_crop(crop: np.ndarray, processor, model) -> tuple:
    """
    Runs TrOCR on a single cropped line image.
    Returns (text, confidence) where confidence is an approximate score
    derived from the average token probability of the generated sequence.
    """
    import torch

    if crop.size == 0:
        return "", 0.0

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(_TORCH_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=64,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Approximate confidence: average of max softmax prob per generated token
    if outputs.scores:
        probs = []
        for step_scores in outputs.scores:
            step_probs = torch.softmax(step_scores, dim=-1)
            probs.append(float(torch.max(step_probs)))
        confidence = sum(probs) / len(probs) if probs else 0.0
    else:
        confidence = 0.0

    return text, confidence


def run_ocr(
    image_path: str,
    apply_preprocessing: bool = True,
    min_confidence: float = 0.60,
    lang: str = "en",
    trocr_model: str = "microsoft/trocr-base-handwritten",
) -> dict:
    """
    Runs the full handwriting OCR pipeline on a single image.

    Returns:
        {
          "text": "combined text, one line per detected text region",
          "lines": [
              {"text": "...", "confidence": 0.91, "box": [[x,y], ...]}
          ],
          "low_confidence_lines": [...],   # flagged for manual review
          "processing_time_ms": 1450,
          "engine": "paddleocr-detect + trocr-recognize"
        }
    """
    start = time.time()

    if apply_preprocessing:
        image = preprocess_image(image_path)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

    tmp_path = "_tmp_preprocessed.jpg"
    cv2.imwrite(tmp_path, image)

    detector = get_detector(lang=lang)
    # Full detect+recognize call (stable, well-tested codepath). We only
    # keep the box coordinates from this — PaddleOCR's own recognized text
    # is discarded, since TrOCR re-reads each crop for handwriting accuracy.
    # (The separate det=True/rec=False "detection-only" mode has a known
    # bug in some PaddleOCR versions that raises a numpy ambiguous-truth-
    # value error internally, so we avoid it entirely.)
    raw_result = detector.ocr(tmp_path, cls=True)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    boxes = _extract_boxes(raw_result)

    processor, model = get_recognizer(trocr_model)

    lines = []
    low_confidence_lines = []
    combined_text_parts = []

    for box in boxes:
        crop = _crop_box(image, box)
        text, confidence = _recognize_crop(crop, processor, model)

        if not text:
            continue

        entry = {"text": text, "confidence": round(confidence, 4), "box": box}
        lines.append(entry)
        combined_text_parts.append(text)
        if confidence < min_confidence:
            low_confidence_lines.append(entry)

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return {
        "text": "\n".join(combined_text_parts),
        "lines": lines,
        "low_confidence_lines": low_confidence_lines,
        "processing_time_ms": elapsed_ms,
        "engine": "paddleocr-detect + trocr-recognize",
    }


def run_ocr_batch(image_paths: list, **kwargs) -> list:
    """Runs run_ocr over a list of image paths, capturing per-image errors
    so one bad file doesn't kill the whole batch."""
    results = []
    for path in image_paths:
        try:
            result = run_ocr(path, **kwargs)
            result["file"] = path
            result["status"] = "ok"
        except Exception as e:
            result = {"file": path, "status": "error", "error": str(e)}
        results.append(result)
    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)

    output = run_ocr(sys.argv[1])
    print(json.dumps(output, indent=2))