"""
Import labeled data from PPOCRLabel-style source folders into dataset/images/.

Source layout (one folder per date, with its own Label.txt that uses paths like `<date>/<file>.jpg`):
    <source_root>/
        20260325/
            img1.jpg
            img2.jpg
            Label.txt         <- lines look like: 20260325/img1.jpg\t[{"transcription":..,"points":..}]
        20260326/
            ...

This script:
  1. Walks each source root (e.g. success/, fail/)
  2. Reads every Label.txt
  3. Copies each referenced image to dataset/images/<prefix>_<date>_<originalname>
  4. Merges all labels into dataset/images/Label.txt with the new flattened paths

Usage:
    python tools/import_labeled_data.py \
        --source C:/Users/andy_ac_chen/success --prefix s \
        --source C:/Users/andy_ac_chen/fail    --prefix f \
        --output dataset/images
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True,
                   help="Source root (e.g. C:/Users/andy_ac_chen/success). May be given multiple times.")
    p.add_argument("--prefix", action="append", required=True,
                   help="Filename prefix for each --source (parallel list, e.g. 's' for success).")
    p.add_argument("--output", default="dataset/images",
                   help="Output directory (default: dataset/images)")
    p.add_argument("--dry_run", action="store_true", help="Print actions without writing files")
    return p.parse_args()


def import_one_source(source_root: Path, prefix: str, out_dir: Path, dry_run: bool):
    if not source_root.is_dir():
        print(f"[SKIP] not a directory: {source_root}")
        return []

    merged = []
    seen_basenames = set()
    for date_dir in sorted(source_root.iterdir()):
        if not date_dir.is_dir():
            continue
        label_path = date_dir / "Label.txt"
        if not label_path.exists():
            print(f"  [skip] no Label.txt in {date_dir.name}")
            continue

        date = date_dir.name
        n_ok = 0
        n_missing = 0
        with label_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    print(f"  [warn] {label_path}:{lineno} malformed line")
                    continue
                rel_path, labels_json = parts
                # rel_path looks like "20260325/img.jpg" — strip the date prefix to get just the file
                rel_posix = rel_path.replace("\\", "/")
                basename = Path(rel_posix).name
                src_file = date_dir / basename
                if not src_file.exists():
                    # try treating rel_path as relative to source_root (one level up)
                    alt = source_root / rel_posix
                    if alt.exists():
                        src_file = alt
                    else:
                        print(f"  [miss] {src_file} not found (from {label_path}:{lineno})")
                        n_missing += 1
                        continue

                new_name = f"{prefix}_{date}_{basename}"
                # guarantee unique within this source
                if new_name in seen_basenames:
                    stem = Path(new_name).stem
                    suf = Path(new_name).suffix
                    i = 1
                    while f"{stem}_{i}{suf}" in seen_basenames:
                        i += 1
                    new_name = f"{stem}_{i}{suf}"
                seen_basenames.add(new_name)

                dest_file = out_dir / new_name
                if not dry_run:
                    shutil.copy2(src_file, dest_file)

                # Validate JSON is parseable
                try:
                    json.loads(labels_json)
                except json.JSONDecodeError as e:
                    print(f"  [warn] {label_path}:{lineno} invalid JSON: {e}")
                    continue

                # new label line: path is "<new_name>" so data_dir can be dataset/images
                merged.append(f"{new_name}\t{labels_json}")
                n_ok += 1
        print(f"  {date}: {n_ok} ok, {n_missing} missing")
    return merged


def main():
    args = parse_args()
    if len(args.source) != len(args.prefix):
        print("ERROR: --source and --prefix must be given the same number of times", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_labels = []
    for source, prefix in zip(args.source, args.prefix):
        print(f"\n=== Importing {source} (prefix='{prefix}') ===")
        labels = import_one_source(Path(source), prefix, out_dir, args.dry_run)
        print(f"  -> {len(labels)} labeled images")
        all_labels.extend(labels)

    merged_path = out_dir / "Label.txt"
    print(f"\nTotal merged label lines: {len(all_labels)}")
    if args.dry_run:
        print(f"[dry_run] would write {merged_path}")
    else:
        with merged_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(all_labels) + "\n")
        print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
