"""
自動重新標記工具 (三層標記策略):
- 用 pretrained v5 det 掃每張圖,自動找出所有文字區域
- 已存在的日期標記 (使用者手工標) → 完整保留 (Positive)
- pretrained 偵測到但不屬於已知日期的區域 → 加入並標成 ### (Ignored)
- 其餘未偵測到的空白區 → 不標 (Background)

用法:
  python auto_relabel.py                               # 處理 success/ + fail/ 底下所有日期
  python auto_relabel.py --dry-run                     # 只看會處理什麼,不寫檔
  python auto_relabel.py --root C:/Users/andy_ac_chen/success
  python auto_relabel.py --date 20260423
  python auto_relabel.py --iou 0.2                     # 比對 IoU 門檻 (預設 0.3)
"""
import sys, io, os, json, argparse, shutil
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, r"C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR")
from ppocr.postprocess.db_postprocess import DBPostProcess


PROJ = Path(r"C:/Users/andy_ac_chen/Desktop/claudeProject")
PRE_ONNX = PROJ / "eval_cpp_runner" / "models" / "ch_PP-OCRv4_det_infer.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None,
                   help="指定只處理某個 root (例如 C:/Users/andy_ac_chen/success),預設兩個都處理")
    p.add_argument("--date", default=None, help="只處理指定日期資料夾")
    p.add_argument("--iou", type=float, default=0.3, help="判定同一框的 IoU 門檻 (預設 0.3)")
    p.add_argument("--dry-run", action="store_true", help="只印訊息不寫檔")
    p.add_argument("--only-new", action="store_true",
                   help="只處理 Label.txt 中還沒有任何標記的新圖 (保留手動修正過的舊圖)")
    return p.parse_args()


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
        return result[0]["points"]


def poly_bbox(pts):
    a = np.asarray(pts)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def load_label_txt(path):
    """讀 PPOCRLabel 格式 Label.txt → dict{key: [annotations]}"""
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        key, js = line.split("\t", 1)
        try:
            anns = json.loads(js)
        except json.JSONDecodeError:
            continue
        d[key] = anns
    return d


def quad_from_pretrained(box):
    """pretrained 出來的是 4x2 numpy array,轉成 int list"""
    return [[int(round(float(x))), int(round(float(y)))] for x, y in box]


def process_folder(folder: Path, det: DetRunner, iou_thr: float, dry_run: bool, only_new: bool = False):
    label_path = folder / "Label.txt"
    images = sorted(folder.glob("*.jpg"))
    if not images:
        print(f"  [skip] 沒有 .jpg: {folder}")
        return

    existing = load_label_txt(label_path)
    # 找出 existing 裡面的檔名 key 前綴 (使用者的 Label.txt 用的相對路徑格式,譬如 "20260420/xxx.jpg")
    # 我們以 folder.name 作為 key prefix,跟原本格式一致
    key_prefix = folder.name

    new_labels = {}
    stats = {"n_img": 0, "n_date": 0, "n_ignore": 0, "n_missed_by_det": 0,
             "n_skip_existing": 0, "n_kept_ignore": 0}

    for img_path in images:
        key = f"{key_prefix}/{img_path.name}"

        # only-new: 已經有標記的圖跳過 (保留手動修正)
        if only_new and key in existing and len(existing[key]) > 0:
            new_labels[key] = existing[key]
            stats["n_skip_existing"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [skip] 讀不到 {img_path.name}")
            continue
        stats["n_img"] += 1

        # 使用者原本標的 (全是日期)
        user_anns = existing.get(key, [])

        # pretrained 偵測
        pre_boxes = det(img)
        pre_bboxes = [poly_bbox(b) for b in pre_boxes]
        user_bboxes = [poly_bbox(a["points"]) for a in user_anns]

        # 過濾 pretrained: 不與任何 user 日期框重疊才保留成 ###
        out_anns = []

        # 1) 先放使用者原本的日期 / 已有的 ### (保留完整 transcription / difficult)
        for a in user_anns:
            out_anns.append({
                "transcription": a.get("transcription", ""),
                "points": [[int(round(float(p[0]))), int(round(float(p[1])))] for p in a["points"]],
                "difficult": a.get("difficult", False),
            })
            if a.get("transcription") == "###":
                stats["n_kept_ignore"] += 1
            else:
                stats["n_date"] += 1

        # 2) 掃 pretrained, 不重疊 user 日期的 → ###
        for pb, pbox in zip(pre_boxes, pre_bboxes):
            overlap = False
            for ub in user_bboxes:
                if bbox_iou(pbox, ub) > iou_thr:
                    overlap = True
                    break
            if overlap:
                continue
            out_anns.append({
                "transcription": "###",
                "points": quad_from_pretrained(pb),
                "difficult": False,
            })
            stats["n_ignore"] += 1

        # 3) 檢查使用者日期裡有幾個 pretrained 沒抓到 (供人工警示)
        for ub in user_bboxes:
            hit = any(bbox_iou(pbox, ub) > iou_thr for pbox in pre_bboxes)
            if not hit:
                stats["n_missed_by_det"] += 1

        new_labels[key] = out_anns

    # 輸出
    lines = []
    for key in sorted(new_labels.keys()):
        anns = new_labels[key]
        lines.append(f"{key}\t{json.dumps(anns, ensure_ascii=False)}")

    if dry_run:
        print(f"  [DRY] {folder.name}: {stats}")
        return stats

    # 備份原 Label.txt
    if label_path.exists():
        bak = label_path.with_suffix(".txt.bak")
        if not bak.exists():
            shutil.copy2(label_path, bak)
            print(f"  [bak] {bak.name}")

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 同時更新 fileState.txt → 把有標記的圖標成「1 (已確認)」
    # PPOCRLabel 會根據這個檔案決定哪些圖開啟時要顯示框
    state_path = folder / "fileState.txt"
    state_lines = []
    for img_path in images:
        key = f"{key_prefix}/{img_path.name}"
        if key in new_labels and len(new_labels[key]) > 0:
            abs_path = str(img_path).replace("/", "\\")
            state_lines.append(f"{abs_path}\t1")
    state_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    msg = (f"  [ok ] {folder.name}: 新處理{stats['n_img']} 跳過已標{stats['n_skip_existing']} "
           f"日期{stats['n_date']} 保留###{stats['n_kept_ignore']} 新增###{stats['n_ignore']} "
           f"(pretrained 漏抓日期={stats['n_missed_by_det']}) → Label.txt + fileState.txt")
    print(msg)
    return stats


def main():
    args = parse_args()

    if args.root:
        roots = [Path(args.root)]
    else:
        roots = [Path(r"C:/Users/andy_ac_chen/success"), Path(r"C:/Users/andy_ac_chen/fail")]

    print(f"=== 載入 pretrained det: {PRE_ONNX.name} ===")
    det = DetRunner(PRE_ONNX)

    total = {"n_img": 0, "n_date": 0, "n_ignore": 0, "n_missed_by_det": 0,
             "n_skip_existing": 0, "n_kept_ignore": 0}
    for root in roots:
        if not root.exists():
            print(f"[warn] root 不存在: {root}")
            continue
        print(f"\n### 處理 root: {root}")
        date_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if args.date:
            date_dirs = [d for d in date_dirs if d.name == args.date]

        for folder in date_dirs:
            s = process_folder(folder, det, args.iou, args.dry_run, args.only_new)
            if s:
                for k in total:
                    total[k] += s[k]

    print(f"\n=== 總計 ===")
    print(f"新處理圖片: {total['n_img']}")
    if args.only_new:
        print(f"跳過已標圖片: {total['n_skip_existing']}")
    print(f"日期標記 (有效): {total['n_date']}")
    print(f"### 保留 (上次跑的): {total['n_kept_ignore']}")
    print(f"### 新增 (本次新偵測): {total['n_ignore']}")
    print(f"pretrained 漏抓的日期: {total['n_missed_by_det']} (這些仍保留在 Label.txt, 但你可能要手動檢查/放大框)")
    if args.dry_run:
        print("(dry-run, 未寫檔)")


if __name__ == "__main__":
    main()
