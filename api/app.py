"""
api/app.py
REST API for the OCR pipeline.

Endpoints:
  POST /predict        - single image -> extracted text + confidence
  POST /predict-batch   - multiple images -> list of results
  GET  /models          - info about the OCR engine in use
  GET  /health          - liveness/readiness check

Run with:
  uvicorn api.app:app --reload --port 8000
"""

import os
import sys
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_pipeline import run_ocr, run_ocr_batch, get_ocr_engine  # noqa: E402

app = FastAPI(
    title="Handwriting / Document OCR API",
    description="Extracts text from images using PaddleOCR, with preprocessing "
                "(denoise, contrast enhancement, deskew) applied first.",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _save_upload(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, upload.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


@app.get("/health")
def health():
    """Basic liveness check + confirms the OCR engine can be loaded."""
    try:
        get_ocr_engine()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {e}"
    return {"status": "ok", "model_status": model_status}


@app.get("/models")
def models():
    return {
        "engine": "PaddleOCR (detection) + TrOCR (handwriting recognition)",
        "detection_model": "PP-OCR text detection (DB-based) - finds text line boxes",
        "recognition_model": "microsoft/trocr-base-handwritten - reads each line, "
                              "fine-tuned on the IAM Handwriting Database",
        "supports_angle_correction": True,
        "notes": "TrOCR handles cursive handwriting far better than PaddleOCR's "
                 "built-in recognizer. Swap to trocr-large-handwritten in "
                 "src/ocr_pipeline.py for a further accuracy bump if needed.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), min_confidence: float = 0.70):
    tmp_path = _save_upload(file)
    try:
        result = run_ocr(tmp_path, min_confidence=min_confidence)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...), min_confidence: float = 0.70):
    tmp_paths = []
    try:
        for f in files:
            tmp_paths.append(_save_upload(f))
        results = run_ocr_batch(tmp_paths, min_confidence=min_confidence)
        return JSONResponse(content={"results": results, "count": len(results)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in tmp_paths:
            shutil.rmtree(os.path.dirname(p), ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
