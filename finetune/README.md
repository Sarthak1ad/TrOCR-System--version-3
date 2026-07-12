# Fine-tuning TrOCR on your own handwriting

Generic pretrained handwriting OCR (PaddleOCR, TrOCR, anything) is tuned to
*average* handwriting. It will always be hit-or-miss on your specific slant,
spacing, and letterforms. Fine-tuning on your own labeled samples is the
actual fix — this teaches the model your handwriting specifically.

## Overview

```
1. Photograph pages of your handwriting        -> finetune/data/raw_photos/
2. Auto-crop into individual lines              -> prepare_dataset.py
3. Label each line with the correct text        -> edit labels.csv by hand
4. Fine-tune TrOCR on your labeled lines         -> finetune_trocr.py
5. Compare before/after accuracy                 -> evaluate.py
6. Point the main app at your fine-tuned model  -> edit src/ocr_pipeline.py
```

## Step 1: Collect handwriting photos

Take clear, well-lit photos of pages of your own handwriting — the more
varied the content (not just repeating the same words), the better the
model generalizes. Put them in:

```
finetune/data/raw_photos/
```

**How much do you need?** Rough guide:
- 20-50 photos of full notebook pages ≈ 300-800 individual lines after
  auto-cropping — enough for a noticeable improvement
- More is better, but there are diminishing returns past ~1000 lines
  unless your handwriting varies a lot page to page

## Step 2: Auto-crop lines from your photos

```bash
python finetune/prepare_dataset.py
```

This reuses the same PaddleOCR detection from the main app to find each
line and saves it as its own image:

```
finetune/data/line_crops/photo1_000.png
finetune/data/line_crops/photo1_001.png
finetune/data/line_crops/photo2_000.png
...
```

It also creates `finetune/data/labels.csv` with a row per line crop, ready
for you to fill in the correct transcription:

```csv
filename,text
photo1_000.png,
photo1_001.png,
```

Run this again any time you add more photos — it only appends new rows,
it won't overwrite text you've already filled in.

## Step 3: Label your data (the actual work)

Open `finetune/data/labels.csv` in Excel, Google Sheets, or any text editor.
For each row, open the matching image in `finetune/data/line_crops/` and
type the exact correct text in the `text` column.

```csv
filename,text
photo1_000.png,What is Prompt Engineering?
photo1_001.png,Prompt engineering involves modifying and augmenting the user
```

**Tips:**
- Transcribe exactly what's written, including your own punctuation/casing
- Skip lines that are illegible even to you — just delete that row
- This is tedious but it's the actual bottleneck for accuracy; there's no
  shortcut here

## Step 4: Fine-tune

Once you have at least ~150-300 labeled lines:

```bash
python finetune/finetune_trocr.py
```

This starts from `microsoft/trocr-base-handwritten` and fine-tunes on your
labeled lines. On CPU this will be slow (hours); on a GPU, much faster
(tens of minutes for a few hundred lines). It automatically:
- Splits your data 90/10 into train/validation
- Trains for 15 epochs, keeping the best checkpoint by validation CER
  (character error rate)
- Saves the final model to `finetune/models/trocr-finetuned/`

## Step 5: Check whether it actually helped

```bash
python finetune/evaluate.py
```

Runs both the original base model and your fine-tuned model on the same
held-out sample and prints predictions side-by-side plus overall CER for
both, so you can see the real improvement rather than guessing.

## Step 6: Use your fine-tuned model in the app

Edit `src/ocr_pipeline.py`:

```python
def get_recognizer(model_name: str = r"C:\path\to\ocr-project\finetune\models\trocr-finetuned"):
```

(or pass `trocr_model=...` explicitly when calling `run_ocr`). Use the
**full absolute path** to avoid ambiguity about the working directory.

Then run the app as usual — it loads your fine-tuned model instead of the
generic one.

## If accuracy still isn't good enough

- Label more data — this usually matters more than any other change
- Make sure your labels are actually correct (typos in labels.csv directly
  hurt training)
- Try more training epochs or a lower learning rate in `finetune_trocr.py`
  if the model seems undertrained (still confusing similar letters)
- Consider starting from `trocr-large-handwritten` instead of `base` if you
  have the compute budget for it
