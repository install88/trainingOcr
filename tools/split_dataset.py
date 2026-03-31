"""
split_dataset.py — 將 PPOCRLabel 輸出的 Label.txt 分割為訓練集和驗證集

PPOCRLabel 輸出格式：
    圖片路徑\t[{"transcription": "文字", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]

用法：
    python tools/split_dataset.py --label_path dataset/Label.txt --output_dir dataset/det --ratio 0.8
"""

import argparse
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="分割 PPOCRLabel 標注檔為訓練集和驗證集")
    parser.add_argument("--label_path", type=str, required=True,
                        help="PPOCRLabel 輸出的 Label.txt 路徑")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="輸出目錄（會自動建立 train/ 和 val/ 子目錄）")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="訓練集比例（預設 0.8，即 80%% 訓練 / 20%% 驗證）")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子（預設 42）")
    parser.add_argument("--copy_images", action="store_true",
                        help="是否將圖片複製到 train/ 和 val/ 目錄下")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.label_path):
        print(f"[錯誤] 找不到標注檔：{args.label_path}")
        return

    with open(args.label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[資訊] 讀取到 {len(lines)} 筆標注")

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * args.ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"[資訊] 訓練集：{len(train_lines)} 筆 | 驗證集：{len(val_lines)} 筆")

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_label_path = os.path.join(args.output_dir, "train_label.txt")
    val_label_path = os.path.join(args.output_dir, "val_label.txt")

    label_dir = os.path.dirname(os.path.abspath(args.label_path))

    def write_labels(lines, label_file, target_dir, copy_images):
        with open(label_file, "w", encoding="utf-8") as f:
            for line in lines:
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    print(f"[警告] 格式不正確，跳過：{line[:80]}")
                    continue

                img_path, annotation = parts
                img_path = img_path.strip()

                if copy_images:
                    abs_img = img_path if os.path.isabs(img_path) else os.path.join(label_dir, img_path)
                    if os.path.exists(abs_img):
                        dst = os.path.join(target_dir, os.path.basename(abs_img))
                        shutil.copy2(abs_img, dst)
                        img_path = os.path.basename(abs_img)
                    else:
                        print(f"[警告] 找不到圖片：{abs_img}")

                f.write(f"{img_path}\t{annotation}\n")

    write_labels(train_lines, train_label_path, train_dir, args.copy_images)
    write_labels(val_lines, val_label_path, val_dir, args.copy_images)

    print(f"[完成] 訓練標注 → {train_label_path}")
    print(f"[完成] 驗證標注 → {val_label_path}")
    if args.copy_images:
        print(f"[完成] 圖片已複製至 {train_dir} 和 {val_dir}")


if __name__ == "__main__":
    main()
