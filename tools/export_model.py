"""
export_model.py — 將訓練好的模型匯出為 Paddle inference 格式

這是呼叫 PaddleOCR 原始碼中 export_model 的包裝腳本，
避免每次都要 cd 到 PaddleOCR 目錄。

用法（Det 模型）：
    python tools/export_model.py --model_type det

用法（Rec 模型）：
    python tools/export_model.py --model_type rec

用法（自訂路徑）：
    python tools/export_model.py \
        --config configs/det/PP-OCRv4_mobile_det_finetune.yml \
        --trained_model output/det/best_accuracy \
        --output_dir output/det/inference
"""

import argparse
import os
import subprocess
import sys

# 專案路徑
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLEOCR_DIR = r"C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR"
EXPORT_SCRIPT = os.path.join(PADDLEOCR_DIR, "tools", "export_model.py")

# 預設設定
DEFAULTS = {
    "det": {
        "config": os.path.join(PROJECT_DIR, "configs", "det", "PP-OCRv4_mobile_det_finetune.yml"),
        "trained_model": os.path.join(PROJECT_DIR, "output", "det", "best_accuracy"),
        "output_dir": os.path.join(PROJECT_DIR, "output", "det", "inference"),
    },
    "rec": {
        "config": os.path.join(PROJECT_DIR, "configs", "rec", "PP-OCRv4_mobile_rec_finetune.yml"),
        "trained_model": os.path.join(PROJECT_DIR, "output", "rec", "best_accuracy"),
        "output_dir": os.path.join(PROJECT_DIR, "output", "rec", "inference"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="匯出訓練好的模型為 Paddle inference 格式")
    parser.add_argument("--model_type", type=str, choices=["det", "rec"],
                        help="模型類型 (det 或 rec)，會自動帶入預設路徑")
    parser.add_argument("--config", type=str, default=None,
                        help="訓練設定檔路徑（覆蓋 model_type 預設值）")
    parser.add_argument("--trained_model", type=str, default=None,
                        help="訓練好的模型路徑（覆蓋 model_type 預設值）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="匯出目錄（覆蓋 model_type 預設值）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_type:
        defaults = DEFAULTS[args.model_type]
        config = args.config or defaults["config"]
        trained_model = args.trained_model or defaults["trained_model"]
        output_dir = args.output_dir or defaults["output_dir"]
    else:
        if not all([args.config, args.trained_model, args.output_dir]):
            print("[錯誤] 若不指定 --model_type，則必須同時提供 --config、--trained_model、--output_dir")
            sys.exit(1)
        config = args.config
        trained_model = args.trained_model
        output_dir = args.output_dir

    if not os.path.exists(EXPORT_SCRIPT):
        print(f"[錯誤] 找不到 PaddleOCR export 腳本：{EXPORT_SCRIPT}")
        sys.exit(1)

    if not os.path.exists(config):
        print(f"[錯誤] 找不到設定檔：{config}")
        sys.exit(1)

    print(f"[資訊] 模型類型：{args.model_type or '自訂'}")
    print(f"[資訊] 設定檔：{config}")
    print(f"[資訊] 訓練模型：{trained_model}")
    print(f"[資訊] 匯出目錄：{output_dir}")
    print()

    cmd = [
        sys.executable,
        EXPORT_SCRIPT,
        "-c", config,
        "-o",
        f"Global.pretrained_model={trained_model}",
        f"Global.save_inference_dir={output_dir}",
    ]

    print(f"[執行] {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=PADDLEOCR_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
