"""
run_ocr.py — 本機 OCR 推論腳本（使用 PaddleOCR 套件）

功能：
  1. 對圖片執行 det（文字偵測）+ rec（文字辨識）
  2. 產生帶紅色框線的視覺化圖片
  3. 輸出 JSON 格式的辨識結果（與 C++ eval_cpp_runner 對比用）

用法：
    # 單張圖片
    python tools/run_ocr.py --input dataset/images/test.jpg

    # 整個資料夾
    python tools/run_ocr.py --input eval_cpp_runner/input/success/

    # 指定輸出目錄
    python tools/run_ocr.py --input eval_cpp_runner/input/ --output_dir output/ocr_results/

    # 只跑偵測（不辨識），用於檢查紅框效果
    python tools/run_ocr.py --input test.jpg --det_only

    # 與 C++ 結果比較
    python tools/run_ocr.py --input eval_cpp_runner/input/success/ --compare eval_cpp_runner/out/eval_results.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Windows 終端機 UTF-8 支援
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def get_image_files(input_path):
    """取得所有圖片檔案路徑"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    elif p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
        # 也搜尋子目錄
        for sub in sorted(p.iterdir()):
            if sub.is_dir():
                for f in sorted(sub.iterdir()):
                    if f.is_file() and f.suffix.lower() in exts:
                        files.append(f)
        return files
    return []


def draw_boxes(image, boxes, texts=None, scores=None, color=(0, 0, 255), thickness=2):
    """
    在圖片上畫偵測框

    boxes: list of polygons, 每個 polygon 是 4 個點 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    vis = image.copy()
    for i, box in enumerate(boxes):
        box = np.array(box, dtype=np.int32)
        cv2.polylines(vis, [box], isClosed=True, color=color, thickness=thickness)

        # 標註文字和分數
        if texts and i < len(texts):
            label = texts[i]
            if scores and i < len(scores):
                label += f" ({scores[i]:.3f})"
            # 文字位置：框的左上角偏上
            x, y = int(box[0][0]), int(box[0][1]) - 5
            y = max(y, 15)
            cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
    return vis


def boxes_to_frame(box):
    """將 4 點 polygon 轉換為 frame 格式（與 C++ 輸出對齊）"""
    box = np.array(box, dtype=np.int32)
    x_min, y_min = box.min(axis=0)
    x_max, y_max = box.max(axis=0)
    return {
        "top": int(y_min),
        "left": int(x_min),
        "width": int(x_max - x_min),
        "height": int(y_max - y_min),
    }


def run_single_image(ocr, image_path, output_dir, det_only=False, save_vis=True):
    """對單張圖片執行 OCR"""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [警告] 無法讀取圖片：{image_path}")
        return None

    start_time = time.time()

    # 執行 OCR
    result = ocr.predict(str(image_path))

    elapsed = time.time() - start_time

    # 解析結果
    rows = []
    boxes = []
    texts = []
    scores = []

    for res in result:
        # PaddleOCR 3.x 回傳 dict-like OCRResult 物件
        # 支援 res['key'] 和 res.key 兩種存取方式
        def get_field(obj, key, default=None):
            """安全取得欄位值（相容 dict / object）"""
            try:
                return obj[key]
            except (KeyError, TypeError):
                return getattr(obj, key, default)

        dt_polys = get_field(res, "dt_polys")
        rec_texts = get_field(res, "rec_texts")
        rec_scores = get_field(res, "rec_scores")

        if dt_polys is not None and len(dt_polys) > 0:
            for i in range(len(dt_polys)):
                box = dt_polys[i].tolist() if hasattr(dt_polys[i], 'tolist') else dt_polys[i]

                text = ""
                score = 0.0
                if not det_only and rec_texts and i < len(rec_texts):
                    text = rec_texts[i]
                    score = float(rec_scores[i]) if rec_scores and i < len(rec_scores) else 0.0

                boxes.append(box)
                texts.append(text)
                scores.append(score)

                row = {
                    "row_text": text,
                    "score": round(score, 6),
                    "frame": boxes_to_frame(box),
                    "polygon": box,
                }
                rows.append(row)

    # 組成輸出 JSON（與 C++ eval_results.jsonl 格式對齊）
    output = {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "rows": rows,
        "elapsed_ms": round(elapsed * 1000, 1),
        "source": "python_paddleocr",
    }

    # 儲存視覺化圖片
    if save_vis and output_dir:
        debug_dir = Path(output_dir) / "vis"
        debug_dir.mkdir(parents=True, exist_ok=True)

        vis_img = draw_boxes(img, boxes, texts if not det_only else None, scores if not det_only else None)
        vis_path = debug_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        output["debug_dir"] = str(debug_dir)

    return output


def load_comparison(jsonl_path):
    """載入 C++ eval_results.jsonl 用於比較"""
    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results[data["image_name"]] = data
                except json.JSONDecodeError as e:
                    print(f"  [警告] JSONL 第 {line_num} 行解析失敗：{e}")
                    continue
    return results


def compare_results(py_result, cpp_result):
    """比較 Python 與 C++ 的 OCR 結果"""
    diffs = []

    py_texts = [r["row_text"] for r in py_result.get("rows", [])]
    cpp_texts = [r["row_text"] for r in cpp_result.get("rows", [])]

    if len(py_texts) != len(cpp_texts):
        diffs.append(f"偵測數量不同：Python={len(py_texts)}, C++={len(cpp_texts)}")

    # 逐行比較
    for i in range(min(len(py_texts), len(cpp_texts))):
        if py_texts[i] != cpp_texts[i]:
            diffs.append(f"  行{i}: Python='{py_texts[i]}' vs C++='{cpp_texts[i]}'")

    return diffs


def parse_args():
    parser = argparse.ArgumentParser(description="本機 OCR 推論（PaddleOCR）")
    parser.add_argument("--input", type=str, required=True,
                        help="輸入圖片路徑或資料夾")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="輸出目錄（預設：output/ocr_results/）")
    parser.add_argument("--det_only", action="store_true",
                        help="只執行偵測（不辨識），用於檢查紅框")
    parser.add_argument("--compare", type=str, default=None,
                        help="C++ eval_results.jsonl 路徑，用於比較結果")
    parser.add_argument("--det_model_dir", type=str, default=None,
                        help="自訂 det 模型路徑（Paddle inference 目錄或 ONNX）")
    parser.add_argument("--rec_model_dir", type=str, default=None,
                        help="自訂 rec 模型路徑（Paddle inference 目錄或 ONNX）")
    parser.add_argument("--no_vis", action="store_true",
                        help="不儲存視覺化圖片")
    parser.add_argument("--ocr_version", type=str, default="PP-OCRv4",
                        help="OCR 版本（預設 PP-OCRv4）")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or str(PROJECT_DIR / "output" / "ocr_results")
    os.makedirs(output_dir, exist_ok=True)

    # 取得圖片清單
    image_files = get_image_files(args.input)
    if not image_files:
        print(f"[錯誤] 找不到圖片：{args.input}")
        sys.exit(1)

    print(f"[資訊] 找到 {len(image_files)} 張圖片")
    print(f"[資訊] 輸出目錄：{output_dir}")

    # 初始化 PaddleOCR
    print("[資訊] 載入 PaddleOCR 模型...")

    from paddleocr import PaddleOCR

    ocr_kwargs = {
        "ocr_version": args.ocr_version,
        "lang": "ch",
    }

    if args.det_model_dir:
        ocr_kwargs["text_detection_model_dir"] = args.det_model_dir
    if args.rec_model_dir:
        ocr_kwargs["text_recognition_model_dir"] = args.rec_model_dir

    ocr = PaddleOCR(**ocr_kwargs)
    print("[資訊] 模型載入完成")

    # 載入 C++ 比較資料（如有）
    cpp_results = None
    if args.compare:
        if os.path.exists(args.compare):
            cpp_results = load_comparison(args.compare)
            print(f"[資訊] 載入 C++ 結果 {len(cpp_results)} 筆，用於比較")
        else:
            print(f"[警告] 找不到比較檔案：{args.compare}")

    # 逐張處理
    all_results = []
    total_time = 0
    diff_count = 0

    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")

        result = run_single_image(
            ocr, img_path, output_dir,
            det_only=args.det_only,
            save_vis=not args.no_vis,
        )

        if result is None:
            continue

        all_results.append(result)
        total_time += result["elapsed_ms"]

        # 輸出摘要
        for row in result["rows"]:
            text = row["row_text"] or "(det only)"
            score = row["score"]
            f = row["frame"]
            print(f"  [{score:.3f}] '{text}' @ ({f['left']},{f['top']}) {f['width']}x{f['height']}")

        print(f"  耗時：{result['elapsed_ms']:.0f}ms")

        # 與 C++ 比較
        if cpp_results and result["image_name"] in cpp_results:
            diffs = compare_results(result, cpp_results[result["image_name"]])
            if diffs:
                diff_count += 1
                print(f"  [差異]")
                for d in diffs:
                    print(f"    {d}")
            else:
                print(f"  [一致] 與 C++ 結果相同")

    # 儲存完整結果
    jsonl_path = os.path.join(output_dir, "python_ocr_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 也存一份易讀的 JSON
    json_path = os.path.join(output_dir, "python_ocr_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 統計
    print(f"\n{'='*60}")
    print(f"[完成] 共處理 {len(all_results)} 張圖片")
    print(f"[完成] 總耗時：{total_time:.0f}ms（平均 {total_time/max(len(all_results),1):.0f}ms/張）")
    print(f"[完成] 結果儲存：{jsonl_path}")
    if not args.no_vis:
        print(f"[完成] 視覺化圖片：{os.path.join(output_dir, 'vis')}")
    if cpp_results:
        print(f"[比較] 有差異：{diff_count}/{len(all_results)} 張")


if __name__ == "__main__":
    main()
