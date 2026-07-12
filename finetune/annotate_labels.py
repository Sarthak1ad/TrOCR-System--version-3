import argparse
import pandas as pd
from PIL import Image
import os
import sys

def annotate(csv_path, images_dir, start=0, save_every=10):
    df = pd.read_csv(csv_path)
    total = len(df)
    uns = df['text'].isna() | (df['text'].str.strip() == '')
    indices = [i for i in range(start, total) if uns.iloc[i]]
    if not indices:
        print('No unlabeled rows found.')
        return
    print(f'Found {len(indices)} unlabeled rows (from {total} total).')
    for count, i in enumerate(indices, 1):
        row = df.iloc[i]
        img_path = row['image'] if 'image' in df.columns else row.get('filename', None)
        if not img_path:
            print(f'Row {i} has no image path column. Skipping.')
            continue
        # resolve path
        img_full = img_path if os.path.isabs(img_path) else os.path.join(images_dir, os.path.basename(img_path))
        if not os.path.exists(img_full):
            print(f"Image not found: {img_full} (row {i}). Skipping.")
            continue
        try:
            im = Image.open(img_full)
            im.show()
        except Exception as e:
            print(f'Error opening image {img_full}: {e}')
            continue
        try:
            txt = input(f'[{count}/{len(indices)}] Enter text for {os.path.basename(img_full)} (empty=skip, ":q"=quit): ').strip()
        except EOFError:
            print('\nInput closed, exiting.')
            break
        if txt == ':q':
            break
        if txt == '':
            print('Skipped.')
            continue
        df.at[i, 'text'] = txt
        if count % save_every == 0:
            df.to_csv(csv_path, index=False)
            print(f'Saved progress to {csv_path} (row {i}).')
    # final save
    df.to_csv(csv_path, index=False)
    print(f'Annotation finished. Saved to {csv_path}.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='finetune/data/labels.csv', help='Path to labels.csv')
    p.add_argument('--images', default='finetune/data/line_crops', help='Directory containing line crop images')
    p.add_argument('--start', type=int, default=0, help='Start index in CSV')
    p.add_argument('--save-every', type=int, default=10, help='Autosave every N annotations')
    args = p.parse_args()
    if not os.path.exists(args.csv):
        print(f'CSV file not found: {args.csv}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.images):
        print(f'Images dir not found: {args.images}', file=sys.stderr)
        sys.exit(1)
    annotate(args.csv, args.images, start=args.start, save_every=args.save_every)
