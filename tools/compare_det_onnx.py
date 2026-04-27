"""
比較預訓練 v5 det vs fine-tuned v5 det (都是 ONNX)
對 success / fail 各抽樣若干張,並排畫出結果 + 產生 HTML 總覽

用法:
  python compare_det_onnx.py                                     # 預設 20260423
  python compare_det_onnx.py --date 20260422                     # 換日期
  python compare_det_onnx.py --success D:/s --fail D:/f --out D:/out --n 20
"""
import sys, io, os, random, argparse
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, r"C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR")
from ppocr.postprocess.db_postprocess import DBPostProcess


PROJ = Path(r"C:/Users/andy_ac_chen/Desktop/claudeProject")
MODELS = PROJ / "eval_cpp_runner" / "models"
PRE_ONNX = MODELS / "ch_PP-OCRv4_det_infer.onnx"
FT_ONNX = MODELS / "ch_PP-OCRv4_det_infer_new.onnx"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="20260423",
                   help="日期資料夾名稱 (預設 20260423)")
    p.add_argument("--success", default=None, help="success 資料夾完整路徑")
    p.add_argument("--fail", default=None, help="fail 資料夾完整路徑")
    p.add_argument("--out", default=None, help="輸出資料夾")
    p.add_argument("--n", type=int, default=10, help="每個類別抽幾張 (預設 10)")
    return p.parse_args()

args = parse_args()
BASE = Path(r"C:/Users/andy_ac_chen")
SUCCESS_DIR = Path(args.success) if args.success else BASE / "success" / args.date
FAIL_DIR    = Path(args.fail)    if args.fail    else BASE / "fail"    / args.date
OUT_DIR     = Path(args.out)     if args.out     else PROJ / "output" / f"compare_det_{args.date}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLE_PER = args.n

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, max_side=960):
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(32, ((nh + 31) // 32) * 32)
    nw = max(32, ((nw + 31) // 32) * 32)
    resized = cv2.resize(img_bgr, (nw, nh))
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    ratio_h = nh / h
    ratio_w = nw / w
    return x, (h, w, ratio_h, ratio_w)


class DetRunner:
    def __init__(self, onnx_path, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.post = DBPostProcess(
            thresh=thresh, box_thresh=box_thresh,
            max_candidates=1000, unclip_ratio=unclip_ratio,
            use_dilation=False, score_mode="fast", box_type="quad",
        )

    def __call__(self, img_bgr):
        x, (h, w, rh, rw) = preprocess(img_bgr)
        out = self.sess.run(None, {self.inp_name: x})[0]
        shape_list = np.array([[h, w, rh, rw]])
        result = self.post({"maps": out}, shape_list)
        boxes = result[0]["points"]
        return boxes


def draw_boxes(img, boxes, color, label):
    img = img.copy()
    for b in boxes:
        pts = np.array(b, np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    cv2.rectangle(img, (0, 0), (260, 32), (255, 255, 255), -1)
    cv2.putText(img, f"{label}: {len(boxes)} box",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def sample_images(folder, n, seed=42):
    random.seed(seed)
    imgs = sorted([p for p in folder.glob("*.jpg")])
    return random.sample(imgs, min(n, len(imgs)))


def main():
    print(f"=== 載入模型 ===")
    det_pre = DetRunner(PRE_ONNX)
    det_ft = DetRunner(FT_ONNX)
    print(f"pretrained: {PRE_ONNX.name}")
    print(f"finetuned:  {FT_ONNX.name}")

    samples = []
    for tag, folder in [("fail", FAIL_DIR), ("success", SUCCESS_DIR)]:
        imgs = sample_images(folder, N_SAMPLE_PER)
        samples.extend([(tag, p) for p in imgs])
        print(f"{tag}: 抽 {len(imgs)} / {len(list(folder.glob('*.jpg')))}")

    summary = []
    for i, (tag, img_path) in enumerate(samples, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] 讀不到 {img_path.name}")
            continue
        boxes_pre = det_pre(img)
        boxes_ft = det_ft(img)
        # box 寬度統計
        def widths(bs):
            ws = []
            for b in bs:
                pts = np.array(b)
                ws.append(pts[:, 0].max() - pts[:, 0].min())
            return ws
        w_pre = widths(boxes_pre)
        w_ft = widths(boxes_ft)
        summary.append({
            "idx": i, "tag": tag, "name": img_path.name,
            "n_pre": len(boxes_pre), "n_ft": len(boxes_ft),
            "avg_w_pre": np.mean(w_pre) if w_pre else 0,
            "avg_w_ft": np.mean(w_ft) if w_ft else 0,
        })
        # 畫並排
        vis_pre = draw_boxes(img, boxes_pre, (0, 200, 0), "PRETRAINED")
        vis_ft = draw_boxes(img, boxes_ft, (0, 0, 255), "FINETUNED")
        side = np.hstack([vis_pre, vis_ft])
        out_name = f"{i:02d}_{tag}_{img_path.stem[:30]}.jpg"
        cv2.imwrite(str(OUT_DIR / out_name), side)
        print(f"  [{i:02d}/{len(samples)}] {tag:7s} pre={len(boxes_pre):2d} ft={len(boxes_ft):2d}  {img_path.name}")

    # HTML 總表
    html = ["<html><head><meta charset='utf-8'><title>Det ONNX 比較</title>",
            "<style>body{font-family:sans-serif;background:#f4f6f9}",
            "table{border-collapse:collapse;margin:20px}",
            "th,td{border:1px solid #ccc;padding:6px;text-align:center}",
            "th{background:#1f3a5f;color:white}",
            ".tag-fail{background:#fbeaea}.tag-succ{background:#e6f4ea}",
            "img{max-width:1200px;border:1px solid #aaa;margin:5px 0}",
            ".row{display:flex;flex-direction:column;margin:20px;background:white;padding:15px;border-radius:6px}",
            ".stats{font-size:14px;color:#555}</style></head><body>"]
    html.append(f"<h1>ONNX det 模型比較</h1>")
    html.append(f"<p><b>pretrained</b>: {PRE_ONNX.name}  vs  <b>finetuned</b>: {FT_ONNX.name}</p>")

    # 統計表
    html.append("<table><tr><th>#</th><th>類別</th><th>檔名</th>"
                "<th>pretrained boxes</th><th>finetuned boxes</th>"
                "<th>pre 平均寬</th><th>ft 平均寬</th><th>寬度覆蓋率</th></tr>")
    for s in summary:
        cov = (s['avg_w_ft'] / s['avg_w_pre'] * 100) if s['avg_w_pre'] > 0 else 0
        cls = "tag-fail" if s['tag'] == "fail" else "tag-succ"
        html.append(f"<tr class='{cls}'>"
                    f"<td>{s['idx']}</td><td>{s['tag']}</td><td>{s['name'][:40]}</td>"
                    f"<td>{s['n_pre']}</td><td>{s['n_ft']}</td>"
                    f"<td>{s['avg_w_pre']:.0f}</td><td>{s['avg_w_ft']:.0f}</td>"
                    f"<td>{cov:.0f}%</td></tr>")
    html.append("</table>")

    # 整體摘要
    avg_pre = np.mean([s['n_pre'] for s in summary])
    avg_ft = np.mean([s['n_ft'] for s in summary])
    html.append(f"<h2>整體摘要</h2>")
    html.append(f"<p>平均 boxes: pretrained = {avg_pre:.1f},  finetuned = {avg_ft:.1f}"
                f" ({(avg_ft/avg_pre*100 if avg_pre else 0):.0f}%)</p>")
    miss = sum(1 for s in summary if s['n_ft'] == 0)
    html.append(f"<p>finetuned 完全抓不到的圖: {miss} / {len(summary)}</p>")

    # 逐張圖
    html.append("<h2>逐張對照 (左: pretrained 綠色框 ‧ 右: finetuned 紅色框)</h2>")
    for s in summary:
        cls = "tag-fail" if s['tag'] == "fail" else "tag-succ"
        fname = f"{s['idx']:02d}_{s['tag']}_{Path(s['name']).stem[:30]}.jpg"
        html.append(f"<div class='row {cls}'>")
        html.append(f"<div class='stats'>#{s['idx']} [{s['tag']}] {s['name']}</div>")
        html.append(f"<div class='stats'>pretrained: {s['n_pre']} boxes  ‧  finetuned: {s['n_ft']} boxes</div>")
        html.append(f"<img src='{fname}'>")
        html.append("</div>")
    html.append("</body></html>")

    html_path = OUT_DIR / "report.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== 完成 ===")
    print(f"HTML 報告: {html_path}")
    print(f"並排圖: {OUT_DIR}/*.jpg")
    print(f"平均 boxes: pretrained={avg_pre:.1f} ft={avg_ft:.1f}  完全漏抓={miss}/{len(summary)}")


if __name__ == "__main__":
    main()
