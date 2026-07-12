"""
finetune/finetune_trocr.py

Fine-tunes TrOCR on your own labeled handwriting samples.

PREREQUISITES:
  1. Run prepare_dataset.py first (or otherwise populate data/line_crops/
     with images and data/labels.csv with filename,text rows)
  2. Fill in the "text" column in data/labels.csv for each line crop
  3. Aim for at least 150-300 labeled lines for a noticeable improvement;
     500+ is better if you have the patience

USAGE:
  python finetune/finetune_trocr.py

OUTPUT:
  finetune/models/trocr-finetuned/   <- fine-tuned model + processor,
                                         ready to be pointed at from
                                         src/ocr_pipeline.py
"""

import os
import sys
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Windows OpenMP conflict workaround (same as main app)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_CROPS_DIR = os.path.join(THIS_DIR, "data", "line_crops")
LABELS_CSV = os.path.join(THIS_DIR, "data", "labels.csv")
OUTPUT_DIR = os.path.join(THIS_DIR, "models", "trocr-finetuned")
CHECKPOINT_DIR = os.path.join(THIS_DIR, "models", "checkpoints")

BASE_MODEL = "microsoft/trocr-base-handwritten"  # starting point for fine-tuning
VAL_SPLIT = 0.1
MAX_LABEL_LENGTH = 64
SEED = 42


class HandwritingLineDataset(Dataset):
    """A single labeled line-crop image + its transcription."""

    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = os.path.join(LINE_CROPS_DIR, row["filename"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=MAX_LABEL_LENGTH,
            truncation=True,
        ).input_ids

        # Pad tokens should be ignored in the loss (-100 is the standard
        # "ignore this position" value for HuggingFace's loss functions)
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def load_labeled_rows():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(
            f"{LABELS_CSV} not found. Run prepare_dataset.py first, "
            "then fill in the text column."
        )

    rows = []
    skipped_empty = 0
    skipped_missing_file = 0

    with open(LABELS_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            filename = row.get("filename", "").strip()

            if not text:
                skipped_empty += 1
                continue

            image_path = os.path.join(LINE_CROPS_DIR, filename)
            if not os.path.exists(image_path):
                skipped_missing_file += 1
                continue

            rows.append({"filename": filename, "text": text})

    print(f"Loaded {len(rows)} labeled line(s).")
    if skipped_empty:
        print(f"Skipped {skipped_empty} row(s) with empty text (not yet labeled).")
    if skipped_missing_file:
        print(f"Skipped {skipped_missing_file} row(s) with missing image files.")

    if len(rows) < 20:
        print(
            "\nWARNING: fewer than 20 labeled examples found. Fine-tuning "
            "on this little data won't help much (and may hurt). Aim for "
            "at least 150-300 labeled lines before training."
        )

    return rows


def character_error_rate(predictions, references):
    """
    Simple CER: total edit distance across all pairs / total reference
    character count. Implemented manually (no extra dependency needed)
    using standard Levenshtein distance.
    """

    def levenshtein(a, b):
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            current_row = [i + 1]
            for j, cb in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    total_edits = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_edits += levenshtein(pred, ref)
        total_chars += max(len(ref), 1)

    return total_edits / total_chars if total_chars else 0.0


def main():
    random.seed(SEED)

    print(f"Loading base model: {BASE_MODEL}")
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    rows = load_labeled_rows()
    random.shuffle(rows)

    val_size = max(1, int(len(rows) * VAL_SPLIT))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    print(f"Train set: {len(train_rows)} lines | Validation set: {len(val_rows)} lines")

    train_dataset = HandwritingLineDataset(train_rows, processor)
    val_dataset = HandwritingLineDataset(val_rows, processor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"Training on: {device}")

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # -100 is the "ignore this position" placeholder used for labels,
        # but some Trainer versions also pad the model's predictions with
        # -100 to match label length. The tokenizer can't decode a negative
        # value, so replace it with the real pad token on BOTH arrays
        # before decoding.
        pred_ids = np.where(np.asarray(pred_ids) != -100, pred_ids, processor.tokenizer.pad_token_id)
        label_ids = np.where(np.asarray(label_ids) != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = character_error_rate(pred_str, label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=15,
        learning_rate=5e-5,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LENGTH,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Resume from the latest checkpoint if one exists, instead of always
    # restarting from scratch — useful since training can get interrupted
    # partway through on a long run.
    resume_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR):
        existing = [
            d for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
        ]
        if existing:
            existing.sort(key=lambda d: int(d.split("-")[1]))
            resume_checkpoint = os.path.join(CHECKPOINT_DIR, existing[-1])
            print(f"Found existing checkpoint, resuming from: {resume_checkpoint}")

    print("\nStarting training...\n")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\nDone. To use the fine-tuned model, point src/ocr_pipeline.py at it:")
    print(f'  def get_recognizer(model_name: str = r"{OUTPUT_DIR}"):')


if __name__ == "__main__":
    main()
