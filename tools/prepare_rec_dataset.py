"""
prepare_rec_dataset.py — 從 det 手動標註產生 rec 訓練資料

資料流：
  dataset/det/train_label.txt (PPOCRLabel 格式：filename\\t[{transcription, points}, ...])
  dataset/det/train/*.jpg
  dataset/det/val_label.txt
  dataset/det/val/*.jpg
        ↓  對每個 polygon 做 perspective warp 產生橫向 crop
  dataset/rec/train/crop_NNNNNN.jpg
  dataset/rec/train_label.txt   (每行：crop_NNNNNN.jpg\\t<transcription>)
  dataset/rec/val/crop_NNNNNN.jpg
  dataset/rec/val_label.txt

規則：
  - polygon 為 4 點：perspective transform 到水平矩形
  - 若裁出來後高 > 1.5*寬：順時針轉 90 度（把直書轉橫書）
  - 跳過：transcription == "###" / 空字串 / 裁出來小於 10x10
  - train/val 與 det 完全同步：同張原圖的 crop 只進同一個 split

用法：
    python tools/prepare_rec_dataset.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent
DET_DIR = PROJECT / "dataset" / "det"
REC_DIR = PROJECT / "dataset" / "rec"

MIN_SIZE = 10  # 裁出來小於此尺寸視為爛 crop 跳過


def get_rotated_crop(img, points):
    """對 4 點 polygon 做 perspective warp；若直書則轉正。"""
    pts = np.array(points, dtype=np.float32)
    w = int(round(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3]),
    )))
    h = int(round(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2]),
    )))
    if w < MIN_SIZE or h < MIN_SIZE:
        return None

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 直書 → 順時針 90 度轉成橫書
    if h >= 1.5 * w:
        crop = np.rot90(crop, k=-1)
    return crop


def process_split(split_name, src_img_dir, src_label_file, out_img_dir, out_label_file):
    out_img_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_crops = 0
    n_skip_ignore = 0
    n_skip_empty = 0
    n_skip_small = 0
    n_skip_read = 0
    label_lines = []

    with open(src_label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                img_name, json_str = line.split("\t", 1)
                boxes = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  [警告] 解析失敗：{line[:60]}... ({e})")
                continue

            img_path = src_img_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                n_skip_read += 1
                print(f"  [警告] 讀不到 {img_path}")
                continue

            n_images += 1
            stem = Path(img_name).stem

            for i, box in enumerate(boxes):
                text = box.get("transcription", "").strip()
                points = box.get("points")
                if text == "###":
                    n_skip_ignore += 1
                    continue
                if not text:
                    n_skip_empty += 1
                    continue
                if not points or len(points) != 4:
                    continue

                crop = get_rotated_crop(img, points)
                if crop is None:
                    n_skip_small += 1
                    continue

                crop_name = f"{stem}_box{i:02d}.jpg"
                cv2.imwrite(str(out_img_dir / crop_name), crop)
                label_lines.append(f"{crop_name}\t{text}")
                n_crops += 1

    with open(out_label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"\n[{split_name}] 完成")
    print(f"  處理原圖：{n_images}")
    print(f"  產生 crop：{n_crops}")
    print(f"  跳過 - ###（忽略）：{n_skip_ignore}")
    print(f"  跳過 - 空文字：{n_skip_empty}")
    print(f"  跳過 - 太小：{n_skip_small}")
    print(f"  跳過 - 讀不到圖：{n_skip_read}")
    print(f"  輸出：{out_img_dir}")
    print(f"  標注：{out_label_file}")

    return n_crops


def main():
    print(f"=== 從 {DET_DIR} 產生 rec 資料集到 {REC_DIR} ===\n")

    total_train = process_split(
        "train",
        DET_DIR / "train",
        DET_DIR / "train_label.txt",
        REC_DIR / "train",
        REC_DIR / "train_label.txt",
    )
    total_val = process_split(
        "val",
        DET_DIR / "val",
        DET_DIR / "val_label.txt",
        REC_DIR / "val",
        REC_DIR / "val_label.txt",
    )

    print(f"\n{'='*60}")
    print(f"總計：train={total_train}、val={total_val}、all={total_train + total_val}")
    print(f"下一步：把 dataset/rec/ 打包成 rec_dataset.zip 上傳到 Google Drive")


if __name__ == "__main__":
    main()
