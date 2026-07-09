"""
preprocessing.py
Cleans up an input image before it's sent to the OCR engine.
Handles: denoising, contrast enhancement, and skew correction.
These steps meaningfully improve recognition accuracy on real-world
photos (phone pics of documents, scanned handwriting, etc.)
"""

import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    """Bilateral filter: removes noise while preserving text edges."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE (adaptive histogram equalization) - boosts local contrast,
    which helps a lot with faded or unevenly lit scans/photos."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def detect_skew_angle(image: np.ndarray) -> float:
    """Estimates the rotation angle of the text using minAreaRect
    over thresholded foreground pixels."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def correct_skew(image: np.ndarray, angle: float = None) -> np.ndarray:
    """Rotates the image to straighten out detected text skew."""
    if angle is None:
        angle = detect_skew_angle(image)

    if abs(angle) < 0.3:  # not worth rotating for tiny angles
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(image_path: str, save_path: str = None) -> np.ndarray:
    """
    Full preprocessing pipeline: load -> denoise -> enhance contrast -> deskew.
    Returns the cleaned image as a numpy array (BGR).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = denoise(image)
    image = enhance_contrast(image)
    image = correct_skew(image)

    if save_path:
        cv2.imwrite(save_path, image)

    return image


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path> [output_path]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "preprocessed.jpg"
    preprocess_image(sys.argv[1], out)
    print(f"Saved preprocessed image to {out}")
