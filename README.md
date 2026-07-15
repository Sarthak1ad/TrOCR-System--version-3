# 📝 OCR Text Extraction System

> A production-ready OCR pipeline for handwritten and printed text extraction using **PaddleOCR** for text detection and **Microsoft TrOCR** for handwriting recognition.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebUI-red)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

This project extracts text from images containing handwritten or printed content.

Unlike traditional OCR systems that rely on a single engine, this project combines two state-of-the-art deep learning models:

* **PaddleOCR** → Detects text regions in the image.
* **Microsoft TrOCR** → Recognizes handwritten text inside each detected region.

This hybrid architecture significantly improves handwriting recognition accuracy compared to traditional OCR engines.

---

# Features

* Handwritten Text Recognition
* Printed Text Recognition
* Automatic Text Detection
* Image Preprocessing
* Confidence Score for every text line
* FastAPI REST API
* Streamlit Web Interface
* Batch Prediction Support
* JSON Export
* CSV Export
* TXT Export
* GPU Acceleration (Automatic)
* CPU Compatible

---

# Sample UI

Replace these with your screenshots.

## Home Screen

```md
![Home](images/home.png)
```

## Upload Image

```md
![Upload](images/upload.png)
```

## OCR Result

```md
![Result](images/result.png)
```

Recommended folder:

```
images/
├── home.png
├── upload.png
├── result.png
├── architecture.png
├── pipeline.png
```

---

# System Architecture

```md
![Architecture](images/architecture.png)
```

Example architecture:

```
               +--------------------+
               |  Input Image       |
               +---------+----------+
                         |
                         ▼
              Image Preprocessing
        (Deskew • Denoise • Contrast)
                         |
                         ▼
                 PaddleOCR Detector
              (Find Text Bounding Boxes)
                         |
                         ▼
             Crop Individual Text Lines
                         |
                         ▼
             Microsoft TrOCR Recognizer
            (Handwriting Recognition)
                         |
                         ▼
            Confidence Score Generation
                         |
                         ▼
             JSON / CSV / TXT Output
                         |
          +--------------+--------------+
          |                             |
          ▼                             ▼
     FastAPI REST API             Streamlit UI
```

---

# Deep Learning Pipeline

```md
![Deep Learning Pipeline](images/pipeline.png)
```

Pipeline

```
Input Image
      │
      ▼
Preprocessing
      │
      ▼
PaddleOCR Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Image Cropping
      │
      ▼
Vision Transformer (Encoder)
      │
      ▼
Transformer Decoder
      │
      ▼
Predicted Text
      │
      ▼
Confidence Score
```

---

# Technology Stack

| Category         | Technology      |
| ---------------- | --------------- |
| Language         | Python          |
| OCR Detection    | PaddleOCR       |
| Recognition      | Microsoft TrOCR |
| Framework        | PyTorch         |
| API              | FastAPI         |
| UI               | Streamlit       |
| Image Processing | OpenCV          |
| Deep Learning    | Transformers    |
| Data Processing  | NumPy           |
| Visualization    | Matplotlib      |

---

# Project Structure

```
ocr-project/
│
├── api/
│   └── app.py
│
├── src/
│   ├── preprocessing.py
│   ├── ocr_pipeline.py
│
├── web/
│   └── app.py
│
├── tests/
│   └── test_pipeline.py
│
├── images/
│   ├── architecture.png
│   ├── pipeline.png
│   ├── home.png
│   ├── upload.png
│   └── result.png
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ocr-project.git

cd ocr-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Model Download

The first execution downloads:

| Model              | Size    |
| ------------------ | ------- |
| PaddleOCR Detector | ~5 MB   |
| TrOCR Base         | ~2.3 GB |

After downloading once, the models are cached locally.

Available TrOCR models:

* microsoft/trocr-small-handwritten
* microsoft/trocr-base-handwritten
* microsoft/trocr-large-handwritten

---

# Usage

## Command Line

```bash
cd src

python ocr_pipeline.py sample.jpg
```

---

## Streamlit Web UI

```bash
streamlit run web/app.py
```

Open

```
http://localhost:8501
```

Features

* Drag & Drop Upload
* OCR Preview
* Confidence Scores
* JSON Export
* CSV Export
* TXT Export

---

## FastAPI

```bash
uvicorn api.app:app --reload --port 8000
```

Open

```
http://localhost:8000/docs
```

Available Endpoints

| Method | Endpoint       | Description            |
| ------ | -------------- | ---------------------- |
| POST   | /predict       | OCR on single image    |
| POST   | /predict-batch | OCR on multiple images |
| GET    | /health        | Health Check           |
| GET    | /models        | Model Information      |

---

# Image Preprocessing

Before OCR begins, every image goes through:

* Grayscale Conversion
* Noise Removal
* Contrast Enhancement
* Adaptive Thresholding
* Deskewing
* Line Normalization

These preprocessing steps improve recognition accuracy on phone images and scanned documents.

---

# OCR Workflow

```
Image
   │
   ▼
Preprocessing
   │
   ▼
Text Detection
   │
   ▼
Bounding Boxes
   │
   ▼
Crop Text Lines
   │
   ▼
TrOCR Recognition
   │
   ▼
Confidence Calculation
   │
   ▼
Structured Output
```

---

# Testing

Unit Tests

```bash
pytest tests/ -v
```

Integration Tests

```bash
pytest tests/ -v -m integration
```

---

# Performance

* Supports CPU and GPU
* Automatic CUDA detection
* Faster inference on NVIDIA GPUs
* Batch prediction support
* High handwriting recognition accuracy

---

# Future Improvements

* ONNX Export
* TensorRT Optimization
* Multi-language OCR
* PDF OCR Support
* Fine-tuned TrOCR Models
* Docker Deployment
* Kubernetes Deployment
* Authentication
* OCR History Dashboard

---

# References

## PaddleOCR

https://github.com/PaddlePaddle/PaddleOCR

## Microsoft TrOCR

https://huggingface.co/microsoft/trocr-base-handwritten

## Hugging Face Transformers

https://huggingface.co/docs/transformers

## PyTorch

https://pytorch.org

## FastAPI

https://fastapi.tiangolo.com

## Streamlit

https://streamlit.io

## IAM Handwriting Dataset

https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

---

# License

This project is intended for educational, research, and production use. Please review the licenses of PaddleOCR, TrOCR, and their respective dependencies before commercial deployment.
