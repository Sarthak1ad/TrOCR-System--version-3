"""
finetune/evaluate.py

Compares the ORIGINAL pretrained TrOCR model against your FINE-TUNED model
on a held-out sample of your own labeled lines, so you can see the actual
accuracy improvement rather than guessing.

USAGE:
  python finetune/evaluate.py
"""

import os
import sys
import csv
import random

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from finetune_trocr import (  # noqa: E402
    LINE_CROPS_DIR,
    LABELS_CSV,
    OUTPUT_DIR,
    BASE_MODEL,
    character_error_rate,
    load_labeled_rows,
)

SAMPLE_SIZE = 20  # how many random lines to show side-by-side


def run_model(model_name, rows, device):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    predictions = []
    for row in rows:
        image = Image.open(os.path.join(LINE_CROPS_DIR, row["filename"])).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        predictions.append(text)

    return predictions


def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"No fine-tuned model found at {OUTPUT_DIR}. Run finetune_trocr.py first.")
        return

    rows = load_labeled_rows()
    random.seed(123)
    random.shuffle(rows)
    sample = rows[:SAMPLE_SIZE]
    references = [r["text"] for r in sample]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Evaluating on {len(sample)} sample lines...\n")

    print("Running BASE model (pretrained, not fine-tuned)...")
    base_predictions = run_model(BASE_MODEL, sample, device)
    base_cer = character_error_rate(base_predictions, references)

    print("Running FINE-TUNED model...")
    finetuned_predictions = run_model(OUTPUT_DIR, sample, device)
    finetuned_cer = character_error_rate(finetuned_predictions, references)

    print("\n" + "=" * 70)
    print(f"{'GROUND TRUTH':<25}{'BASE MODEL':<25}{'FINE-TUNED':<25}")
    print("=" * 70)
    for ref, base_pred, ft_pred in zip(references, base_predictions, finetuned_predictions):
        print(f"{ref[:23]:<25}{base_pred[:23]:<25}{ft_pred[:23]:<25}")

    print("\n" + "=" * 70)
    print(f"Base model CER:       {base_cer:.2%}  (lower is better)")
    print(f"Fine-tuned model CER: {finetuned_cer:.2%}  (lower is better)")
    improvement = (base_cer - finetuned_cer) / base_cer * 100 if base_cer > 0 else 0
    print(f"Relative improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
