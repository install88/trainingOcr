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
REC_PRE_ONNX = MODELS / "ch_PP-OCRv4_rec_infer.onnx"
REC_FT_ONNX = MODELS / "ch_PP-OCRv4_rec_infer_new.onnx"
REC_DICT = Path(r"C:/Users/andy_ac_chen/Desktop/tool/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt")

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
    def __init__(self, onnx_path, thresh=0.3, box_thresh=0.4, unclip_ratio=2.0, use_dilation=True):
        # thresh 0.3 (對齊 config PostProcess), box_thresh 0.6→0.4 (救低信心日期),
        # unclip_ratio 2.0 (對齊 config PostProcess, 重訓後框自然變大不需要 2.5),
        # use_dilation True (橋接 dot/space 造成的碎片化)
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.post = DBPostProcess(
            thresh=thresh, box_thresh=box_thresh,
            max_candidates=1000, unclip_ratio=unclip_ratio,
            use_dilation=use_dilation, score_mode="fast", box_type="quad",
        )

    def __call__(self, img_bgr):
        x, (h, w, rh, rw) = preprocess(img_bgr)
        out = self.sess.run(None, {self.inp_name: x})[0]
        shape_list = np.array([[h, w, rh, rw]])
        result = self.post({"maps": out}, shape_list)
        boxes = result[0]["points"]
        return boxes


class RecRunner:
    """PP-OCRv4 rec ONNX 推論 (CTC decode)"""
    def __init__(self, onnx_path, dict_path, target_h=48, target_w=320):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.target_h = target_h
        self.target_w = target_w
        # PaddleOCR CTC dict: 0=blank, 1..N=chars, N+1=space
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        self.chars = ["<blank>"] + chars + [" "]

    def crop_quad(self, img, box):
        pts = np.array(box, dtype=np.float32)
        w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        h = max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0]))
        w, h = int(round(w)), int(round(h))
        if w < 5 or h < 5:
            return None
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img, M, (w, h))
        # 直立的字 → 旋轉 90
        if h * 1.0 / max(w, 1) >= 1.5:
            crop = np.rot90(crop)
        return crop

    def preprocess(self, crop):
        h, w = crop.shape[:2]
        ratio = w / max(h, 1)
        new_w = min(int(self.target_h * ratio), self.target_w)
        new_w = max(new_w, 1)
        resized = cv2.resize(crop, (new_w, self.target_h))
        x = resized.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        pad = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
        pad[:, :new_w, :] = x
        return pad.transpose(2, 0, 1)[None]

    def ctc_decode(self, preds, probs):
        prev = -1
        out = []
        scores = []
        for i, p in enumerate(preds):
            if p != prev and p != 0:
                if p < len(self.chars):
                    out.append(self.chars[p])
                    scores.append(float(probs[i]))
            prev = p
        text = "".join(out)
        score = float(np.mean(scores)) if scores else 0.0
        return text, score

    def __call__(self, img, boxes):
        results = []
        for box in boxes:
            try:
                crop = self.crop_quad(img, box)
                if crop is None:
                    results.append(("", 0.0))
                    continue
                x = self.preprocess(crop)
                out = self.sess.run(None, {self.inp_name: x})[0][0]
                preds = np.argmax(out, axis=-1)
                probs = np.max(out, axis=-1)
                text, score = self.ctc_decode(preds, probs)
                results.append((text, score))
            except Exception as e:
                results.append((f"[err:{e}]", 0.0))
        return results


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
    rec_pre_m = RecRunner(REC_PRE_ONNX, REC_DICT)
    rec_ft_m = RecRunner(REC_FT_ONNX, REC_DICT)
    print(f"det pretrained: {PRE_ONNX.name}")
    print(f"det finetuned:  {FT_ONNX.name}")
    print(f"rec pretrained: {REC_PRE_ONNX.name}")
    print(f"rec finetuned:  {REC_FT_ONNX.name}")

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
        # 對每個 det 結果都跑兩個 rec 模型
        rec_pre_on_pre = rec_pre_m(img, boxes_pre)   # det pretrained + rec pretrained
        rec_ft_on_pre  = rec_ft_m(img, boxes_pre)    # det pretrained + rec finetuned
        rec_pre_on_ft  = rec_pre_m(img, boxes_ft)    # det finetuned  + rec pretrained
        rec_ft_on_ft   = rec_ft_m(img, boxes_ft)     # det finetuned  + rec finetuned
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
            "rec_pre_on_pre": rec_pre_on_pre,
            "rec_ft_on_pre":  rec_ft_on_pre,
            "rec_pre_on_ft":  rec_pre_on_ft,
            "rec_ft_on_ft":   rec_ft_on_ft,
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
            ".row{display:flex;flex-direction:column;margin:20px;background:white;padding:15px;border-radius:6px}",
            ".stats{font-size:14px;color:#555;margin-bottom:8px}",
            ".main{display:flex;gap:15px;align-items:flex-start}",
            ".main img{flex:1;max-width:1100px;border:1px solid #aaa;height:auto}",
            ".rec-panel{flex:0 0 540px;font-size:13px;display:flex;flex-direction:column;gap:8px}",
            ".rec-block{background:#fafafa;border:1px solid #ddd;border-radius:4px;padding:8px;max-height:380px;overflow-y:auto}",
            ".rec-block h4{margin:0 0 6px;font-size:13px}",
            ".rec-block .pre{color:#0a7d2c}.rec-block .ft{color:#c0392b}",
            ".rec-row{display:flex;gap:6px;border-bottom:1px dashed #eee;padding:2px 0}",
            ".rec-row .col{flex:1;font-family:Consolas,monospace;font-size:12px;padding:2px 4px;overflow-wrap:anywhere}",
            ".rec-row .col-old{background:#eef5ff}.rec-row .col-new{background:#fff3e0}",
            ".rec-row .score{color:#888;font-size:10px;margin-left:4px}",
            ".rec-head{display:flex;gap:6px;font-size:11px;font-weight:bold;color:#444;margin-bottom:3px}",
            ".rec-head .col{flex:1;text-align:center;padding:2px}",
            ".date-hit{background:#fff3a8;font-weight:bold}",
            "</style></head><body>"]
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
    import re
    DATE_RE = re.compile(r"(20\d{2})[\.\-/年]?(\d{1,2})[\.\-/月]?(\d{1,2})|"
                         r"(\d{1,2})[\.\-/](\d{1,2})[\.\-/](20\d{2}|\d{2})|"
                         r"\d{8}")

    def esc(s):
        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def fmt_rec_pair(rec_old, rec_new):
        """並排顯示舊 rec / 新 rec"""
        if not rec_old:
            return "<div class='rec-row'><div class='col'>(無 box)</div></div>"
        head = ("<div class='rec-head'>"
                "<div class='col'>rec_pretrained</div>"
                "<div class='col'>rec_finetuned</div></div>")
        rows = [head]
        for (t_o, s_o), (t_n, s_n) in zip(rec_old, rec_new):
            o_disp = esc(t_o) if t_o else "(空)"
            n_disp = esc(t_n) if t_n else "(空)"
            o_hit = "date-hit" if t_o and DATE_RE.search(t_o) else ""
            n_hit = "date-hit" if t_n and DATE_RE.search(t_n) else ""
            rows.append(
                f"<div class='rec-row'>"
                f"<div class='col col-old {o_hit}'>{o_disp}<span class='score'>{s_o:.2f}</span></div>"
                f"<div class='col col-new {n_hit}'>{n_disp}<span class='score'>{s_n:.2f}</span></div>"
                f"</div>"
            )
        return "".join(rows)

    html.append("<h2>逐張對照 (左: det pretrained 綠框 ‧ 右: det finetuned 紅框)</h2>")
    html.append("<p style='margin-left:20px;font-size:13px;color:#555'>"
                "右側每個 det 框並排顯示 <b>rec_pretrained</b>（藍底）和 <b>rec_finetuned</b>（橘底）的辨識結果。"
                "<span class='date-hit'>黃色標記</span>表示文字符合日期 regex</p>")
    for s in summary:
        cls = "tag-fail" if s['tag'] == "fail" else "tag-succ"
        fname = f"{s['idx']:02d}_{s['tag']}_{Path(s['name']).stem[:30]}.jpg"
        html.append(f"<div class='row {cls}'>")
        html.append(f"<div class='stats'>#{s['idx']} [{s['tag']}] {s['name']}</div>")
        html.append(f"<div class='stats'>det pretrained: {s['n_pre']} boxes  ‧  det finetuned: {s['n_ft']} boxes</div>")
        html.append("<div class='main'>")
        html.append(f"<img src='{fname}'>")
        html.append("<div class='rec-panel'>")
        html.append(f"<div class='rec-block'><h4 class='pre'>DET PRETRAINED ({s['n_pre']})</h4>"
                    f"{fmt_rec_pair(s['rec_pre_on_pre'], s['rec_ft_on_pre'])}</div>")
        html.append(f"<div class='rec-block'><h4 class='ft'>DET FINETUNED ({s['n_ft']})</h4>"
                    f"{fmt_rec_pair(s['rec_pre_on_ft'], s['rec_ft_on_ft'])}</div>")
        html.append("</div>")  # rec-panel
        html.append("</div>")  # main
        html.append("</div>")  # row
    html.append("</body></html>")

    html_path = OUT_DIR / "report.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== 完成 ===")
    print(f"HTML 報告: {html_path}")
    print(f"並排圖: {OUT_DIR}/*.jpg")
    print(f"平均 boxes: pretrained={avg_pre:.1f} ft={avg_ft:.1f}  完全漏抓={miss}/{len(summary)}")


if __name__ == "__main__":
    main()
