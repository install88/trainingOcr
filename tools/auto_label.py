"""
Auto-label images using PaddleOCR with full labeling rules applied.

Rules implemented:
  - Keep: EXP / MFG / shelf-life / date-value boxes
  - Remove: storage instructions, lot codes, timestamps, irrelevant text
  - Clean: trailing lot/batch/time codes from transcription
  - Priority: if direct EXP exists → drop MFG and pure shelf-life boxes
  - Header-only (BEST BEFORE / 有效日期 without date value) → discard

Usage:
    python tools/auto_label.py --img_dir C:/Users/andy_ac_chen/success/20260419
"""
from __future__ import annotations
import argparse, json, os, re, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────
_MONTH = r'(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'

# A string that actually contains a date VALUE (not just a label)
DATE_VALUE_RE = re.compile(
    r'(?:'
    r'\d{8}|'                                           # YYYYMMDD / DDMMYYYY
    r'\d{2,4}[./\-年]\d{1,2}[./\-月]\d{0,4}|'         # YYYY.MM.DD or YYYY年MM月DD日
    r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|'             # DD/MM/YYYY
    r'\d{1,2}[-./]\d{4}|'                              # MM-YYYY
    r'\d{4}[-./]\d{1,2}(?!\d)|'                        # YYYY-MM
    r'\d{2}\s+\d{2}\s+\d{4}|'                         # DD MM YYYY (space separated)
    r'\d{1,2}\s*' + _MONTH + r'\s*\d{2,4}|'           # 21JAN2027
    r'\d{2,4}\s*' + _MONTH + r'\s*\d{1,2}|'           # 2027JAN21
    r'西元\d{4}年\d{1,2}月|'                           # 西元YYYY年MM月
    r'\d{4}年\d{1,2}月|'                               # YYYY年MM月
    r'(?:保存|有效期間|保存期限)[：:]\s*\d+\s*[年月天日週周]|'  # 有效期間:3年  保存30天
    r'\d+\s*[年月天日週周]'                             # standalone duration: 3年, 30天
    r')',
    re.IGNORECASE | re.UNICODE
)

# EXP-type prefix (direct valid date)
EXP_PREFIX_RE = re.compile(
    r'(?:E[A-Z]{0,2}[P]?\s*[:\.]?\s*|'               # EXP, EYF, EP, EXF misreads
    r'BB[Ff]?\s*[:\.]?\s*|'                           # BB, BBF
    r'BEST\s*BEFORE\s*[:\.]?\s*|'
    r'Expiry|Expiration|USE\s*BY|USEBY|'
    r'有效日期\s*[：:。]?\s*|'
    r'賞味期限\s*[：:。]?\s*|'
    r'消費期限\s*[：:。]?\s*|'
    r'味期限'
    r')',
    re.IGNORECASE | re.UNICODE
)

# MFD-type prefix (manufacture date)
MFD_PREFIX_RE = re.compile(
    r'(?:M[FPG]D?\s*[:\.]?\s*|'                       # MFG, MFD, MPD, MD
    r'PROD?\.?\s*[:\.]?\s*|'                          # PROD, PRD, PR
    r'P[RT][.:]\s*|'
    r'製造日期\s*[：:。]?\s*|'
    r'生產日期\s*[：:。]?\s*|'
    r'有做日期|有料日期|'                              # common OCR misreads of 製造
    r'有製日期'
    r')',
    re.IGNORECASE | re.UNICODE
)

# Shelf-life prefix
SHELF_PREFIX_RE = re.compile(
    r'(?:有效期間|保存期限|保存日期|有効期限)',
    re.UNICODE
)

# Texts to ALWAYS exclude (not date info at all)
EXCLUDE_RE = re.compile(
    r'(?:'
    r'^保存方法|^保存條件|^保存及|^建議食用|^建議烹調|'
    r'消費者服務|服務專線|客服|'
    r'Netweight|net\s*weight|Netto|'
    r'公克|毫升|oz\b|'
    r'^成分|^營養|^Nutrition|^Ingredient|^INGREDIENT|'
    r'條碼|barcode|'
    r'^\d{1,2}:\d{2}(:\d{2})?$|'          # pure timestamp HH:MM
    r'^\d{6}[A-Za-z]\d{3,}$|'             # 250121H001, 240726H001
    r'^[A-Za-z]{1,3}\d{5,}[A-Za-z]\d+$|'  # L43302H011 (letter+digits+letter+digits)
    r'^[A-Za-z]{1,2}\d{2,4}$'             # short lot: R415, A7 (no date pattern)
    r')',
    re.IGNORECASE | re.UNICODE
)

# Trailing codes to strip from transcription
_TRAILING = [
    (re.compile(r'\s+L\d{3,}\s*$'),              ''),  # L096, L103, L110
    (re.compile(r'\s+[A-Z]\d{4,}[A-Z]?\d*\s*$'), ''), # H001, AM20, DC00
    (re.compile(r'[-\s]\d{2}:\d{2}(:\d{2})?\s*$'), ''), # -13:04  01:58
    (re.compile(r'\s+[A-Z]{2}\d{2,3}\s*$'),      ''),  # AD02 BD01
    (re.compile(r'\s+[A-Z]{2,4}\d*\s*$'),        ''),  # AM DC YB LFL
    (re.compile(r'\s*-\d{4,}$'),                 ''),  # trailing -10854 lot codes
    (re.compile(r'\s+\d{5,}\s*$'),              ''),  # trailing 5+ digit lot (14207)
    (re.compile(r'[A-Za-z]{2}\d{2}$'),          ''),  # trailing BD01, AD02
    (re.compile(r'\s+LM\d+\s*$'),               ''),  # trailing LM173
]

def clean_trailing(text: str) -> str:
    for pattern, repl, *cond in _TRAILING:
        min_len = cond[0] if cond else 0
        if len(text) >= min_len:
            text = pattern.sub(repl, text).strip()
    return text

def int_pt(p):
    return [int(round(float(p[0]))), int(round(float(p[1])))]

# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────
def classify(raw_text: str) -> tuple[str, str]:
    """
    Returns (category, cleaned_text)
    category: 'exp' | 'mfd' | 'shelf' | 'value' | 'junk'
    """
    # first clean trailing garbage
    text = clean_trailing(raw_text)

    # hard exclude
    if EXCLUDE_RE.search(text):
        return 'junk', text

    has_date = bool(DATE_VALUE_RE.search(text))

    if EXP_PREFIX_RE.search(text):
        return ('exp' if has_date else 'junk'), text   # header-only → junk

    if MFD_PREFIX_RE.search(text):
        return ('mfd' if has_date else 'junk'), text

    if SHELF_PREFIX_RE.search(text):
        return ('shelf' if has_date else 'junk'), text

    if has_date:
        return 'value', text

    return 'junk', text


def select_boxes(classified: list[tuple[str, str, list]]) -> list[dict]:
    """Apply EXP-priority rule and return final annotation dicts."""
    by_cat = {c: [] for c in ('exp', 'mfd', 'shelf', 'value', 'junk')}
    for cat, txt, poly in classified:
        by_cat[cat].append((txt, poly))

    kept = []

    if by_cat['exp']:
        # direct EXP → only EXP boxes
        kept = by_cat['exp']
    elif by_cat['value']:
        # no EXP prefix, but plain date values
        kept = by_cat['value']
    elif by_cat['mfd'] and (by_cat['shelf'] or by_cat['value']):
        # MFD + shelf-life (for derived EXP)
        kept = by_cat['mfd'] + by_cat['shelf']
    elif by_cat['mfd']:
        kept = by_cat['mfd']
    elif by_cat['shelf']:
        kept = by_cat['shelf']

    return [{"transcription": txt, "points": [int_pt(p) for p in poly],
             "difficult": False} for txt, poly in kept]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--min_score', type=float, default=0.55)
    args = ap.parse_args()

    img_dir = os.path.abspath(args.img_dir)
    out_path = args.out or os.path.join(img_dir, 'Label.txt')
    folder_name = os.path.basename(img_dir)

    print("Loading PaddleOCR …")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    print("OK\n")

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted(f for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in exts)
    print(f"Found {len(images)} images …\n")

    stats = {'ok': 0, 'fallback': 0, 'skip': 0}
    label_lines = []

    for idx, fname in enumerate(images, 1):
        img_path = os.path.join(img_dir, fname)
        rel_path = f"{folder_name}/{fname}"

        try:
            result = ocr.ocr(img_path, cls=True)
        except Exception as e:
            print(f"  [{idx:3d}/{len(images)}] ERROR   {fname}: {e}")
            stats['skip'] += 1
            continue

        page = result[0] if result else []
        if not page:
            print(f"  [{idx:3d}/{len(images)}] NO-DET  {fname}")
            stats['skip'] += 1
            continue

        # classify every detected box
        classified = []
        for line in page:
            poly, (raw_txt, score) = line
            if score < args.min_score:
                continue
            cat, txt = classify(raw_txt)
            classified.append((cat, txt, poly))

        boxes = select_boxes(classified)

        if boxes:
            stats['ok'] += 1
            flag = f"{len(boxes)} box"
            mark = False
        else:
            # fallback: best-scoring non-junk raw box, mark difficult
            candidates = [(score, raw_txt, poly) for line in page
                          for poly, (raw_txt, score) in [line]
                          if score >= args.min_score]
            if candidates:
                candidates.sort(reverse=True)
                _, best_txt, best_poly = candidates[0]
                _, cleaned = classify(best_txt)   # at least clean trailing
                boxes = [{"transcription": cleaned,
                           "points": [int_pt(p) for p in best_poly],
                           "difficult": True}]
                flag = "FALLBACK"
                mark = True
                stats['fallback'] += 1
            else:
                print(f"  [{idx:3d}/{len(images)}] SKIP    {fname}")
                stats['skip'] += 1
                continue

        label_lines.append(f"{rel_path}\t{json.dumps(boxes, ensure_ascii=False)}")
        detail = " | ".join(
            f"{'⚠' if b['difficult'] else '✓'} {b['transcription']}" for b in boxes)
        print(f"  [{idx:3d}/{len(images)}] {flag:8s}  {detail}")

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(label_lines) + '\n')

    print(f"\n{'='*65}")
    print(f"Label.txt → {out_path}")
    print(f"  ✓ date found  : {stats['ok']}")
    print(f"  ⚠ FALLBACK    : {stats['fallback']}  ← 請在 PPOCRLabel 確認紅框")
    print(f"  ✗ skipped     : {stats['skip']}")

if __name__ == '__main__':
    main()
