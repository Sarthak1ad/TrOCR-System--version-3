"""
tests/test_pipeline.py
Basic unit + integration tests for preprocessing and the OCR pipeline.

Run with:
  pytest tests/ -v
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import denoise, enhance_contrast, correct_skew, detect_skew_angle  # noqa: E402


@pytest.fixture
def sample_image():
    """A synthetic image with some text-like rectangles, for testing
    preprocessing without needing a real handwriting sample on disk."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST 123", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


def test_denoise_preserves_shape(sample_image):
    result = denoise(sample_image)
    assert result.shape == sample_image.shape


def test_enhance_contrast_output_shape(sample_image):
    result = enhance_contrast(sample_image)
    assert result.shape[:2] == sample_image.shape[:2]


def test_skew_angle_on_straight_text(sample_image):
    angle = detect_skew_angle(sample_image)
    # Straight, horizontal text should have a small detected angle
    assert abs(angle) < 10


def test_correct_skew_returns_same_size(sample_image):
    result = correct_skew(sample_image)
    assert result.shape == sample_image.shape


def test_correct_skew_no_rotation_for_tiny_angle(sample_image):
    result = correct_skew(sample_image, angle=0.1)
    # angle below threshold -> should return the image unchanged
    assert np.array_equal(result, sample_image)


# --- Integration tests (require paddleocr installed + model download) ---
# These are separated out and skipped by default since they need network
# access to download PaddleOCR's model weights on first run.

@pytest.mark.integration
def test_run_ocr_on_sample_image(tmp_path, sample_image):
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from ocr_pipeline import run_ocr

    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), sample_image)

    result = run_ocr(str(img_path))
    assert "text" in result
    assert "processing_time_ms" in result
    assert isinstance(result["lines"], list)
