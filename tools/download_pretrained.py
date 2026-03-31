"""
download_pretrained.py — 下載 PP-OCRv4 mobile 預訓練模型

下載來源為 PaddleOCR 官方 GitHub Release。

用法：
    python tools/download_pretrained.py --model det
    python tools/download_pretrained.py --model rec
    python tools/download_pretrained.py --model all
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import zipfile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(PROJECT_DIR, "pretrained_models")

# PP-OCRv4 mobile 預訓練模型下載連結
MODELS = {
    "det": {
        "name": "PP-OCRv4 Mobile Det（文字偵測 backbone）",
        "url": "https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams",
        "type": "single_file",
        "filename": "PPLCNetV3_x0_75_ocr_det.pdparams",
    },
    "det_train": {
        "name": "PP-OCRv4 Mobile Det Train（完整偵測模型，含 head）",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar",
        "type": "tar",
        "filename": "ch_PP-OCRv4_det_train.tar",
    },
    "rec": {
        "name": "PP-OCRv4 Mobile Rec Train（文字辨識訓練模型）",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar",
        "type": "tar",
        "filename": "ch_PP-OCRv4_rec_train.tar",
    },
    "det_infer": {
        "name": "PP-OCRv4 Mobile Det Inference（推論模型，可直接轉 ONNX）",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
        "type": "tar",
        "filename": "ch_PP-OCRv4_det_infer.tar",
    },
    "rec_infer": {
        "name": "PP-OCRv4 Mobile Rec Inference（推論模型，可直接轉 ONNX）",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
        "type": "tar",
        "filename": "ch_PP-OCRv4_rec_infer.tar",
    },
}


def download_file(url, dest_path):
    """下載檔案，顯示進度"""
    print(f"  下載中：{url}")
    print(f"  目標：{dest_path}")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  進度：{percent}%% ({mb_done:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  [錯誤] 下載失敗：{e}")
        return False


def extract_tar(filepath, dest_dir):
    """解壓 tar 檔案"""
    print(f"  解壓中：{filepath}")
    try:
        with tarfile.open(filepath, "r:*") as tar:
            tar.extractall(path=dest_dir)
        print(f"  解壓完成")
        return True
    except Exception as e:
        print(f"  [錯誤] 解壓失敗：{e}")
        return False


def extract_zip(filepath, dest_dir):
    """解壓 zip 檔案"""
    print(f"  解壓中：{filepath}")
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(path=dest_dir)
        print(f"  解壓完成")
        return True
    except Exception as e:
        print(f"  [錯誤] 解壓失敗：{e}")
        return False


def download_model(model_key):
    """下載並解壓指定模型"""
    if model_key not in MODELS:
        print(f"[錯誤] 未知的模型：{model_key}")
        return False

    model = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"[下載] {model['name']}")
    print(f"{'='*60}")

    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    dest_path = os.path.join(PRETRAINED_DIR, model["filename"])

    if model["type"] == "single_file":
        if os.path.exists(dest_path):
            print(f"  [跳過] 檔案已存在：{dest_path}")
            return True
        return download_file(model["url"], dest_path)
    else:
        # tar/zip
        extracted_name = model["filename"].replace(".tar", "").replace(".zip", "")
        extracted_dir = os.path.join(PRETRAINED_DIR, extracted_name)

        if os.path.isdir(extracted_dir):
            print(f"  [跳過] 已解壓過：{extracted_dir}")
            return True

        if not os.path.exists(dest_path):
            if not download_file(model["url"], dest_path):
                return False

        if model["type"] == "tar":
            success = extract_tar(dest_path, PRETRAINED_DIR)
        else:
            success = extract_zip(dest_path, PRETRAINED_DIR)

        if success:
            os.remove(dest_path)
            print(f"  已刪除壓縮檔：{dest_path}")

        return success


def parse_args():
    parser = argparse.ArgumentParser(description="下載 PP-OCRv4 mobile 預訓練模型")
    parser.add_argument("--model", type=str, required=True,
                        choices=["det", "det_train", "rec", "det_infer", "rec_infer", "all", "train_all"],
                        help="要下載的模型（det=det backbone, det_train=完整 det 訓練模型, "
                             "rec=rec 訓練模型, all=全部, train_all=訓練用模型）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可下載的模型")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("可下載的模型：")
        for key, model in MODELS.items():
            print(f"  {key:15s} — {model['name']}")
        return

    if args.model == "all":
        keys = list(MODELS.keys())
    elif args.model == "train_all":
        keys = ["det", "det_train", "rec"]
    else:
        keys = [args.model]

    results = {}
    for key in keys:
        results[key] = download_model(key)

    print(f"\n{'='*60}")
    print("[結果]")
    for key, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  {key:15s} — {status}")

    print(f"\n預訓練模型目錄：{PRETRAINED_DIR}")

    if all(results.values()):
        print("\n[提示] 所有模型下載完成！接下來可以開始訓練：")
        print(f"  cd {os.path.join(PROJECT_DIR, '..', 'tool', 'PaddleOCR')}")
        print(f"  python tools/train.py -c {os.path.join(PROJECT_DIR, 'configs', 'det', 'PP-OCRv4_mobile_det_finetune.yml')}")


if __name__ == "__main__":
    main()
