"""
修改 colab_vs_gcp成本.pptx → colab_vs_gcp成本_v2.pptx
目標：讓老闆同意買 Colab Pro+

變更：
  1. Slide 2：建議語改為 Pro+
  2. 插入新 Slide 3：本次訓練規模（圖片量 / 訓練時間 / CU 消耗 / 為何 Pro 不夠）
  3. Slide 5（原 Slide 4 結論）：主要方案從 Pro 改為 Pro+
"""
import sys, copy
sys.stdout.reconfigure(encoding='utf-8')

from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from lxml import etree

INPUT  = 'colab_vs_gcp成本_v8.pptx'   # 以使用者手動改好的 v8 為 base
OUTPUT = 'colab_vs_gcp成本_v9.pptx'

prs = Presentation(INPUT)


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def set_first_run(shape, new_text):
    """
    把 shape 文字框完全清空，只保留第一個 run 的格式，填入 new_text。
    （直接操作 XML，確保舊的多餘 run / 段落全部清除）
    """
    txBody = shape.text_frame._txBody
    paras  = txBody.findall(qn('a:p'))

    # 刪掉多餘段落（只留第一段）
    for p in paras[1:]:
        txBody.remove(p)

    first_p = txBody.findall(qn('a:p'))[0]
    runs    = first_p.findall(qn('a:r'))

    if runs:
        # 保留第一個 run 的格式，更新文字
        first_r = runs[0]
        t_el = first_r.find(qn('a:t'))
        if t_el is None:
            t_el = etree.SubElement(first_r, qn('a:t'))
        t_el.text = new_text
        # 刪掉多餘 run
        for r in runs[1:]:
            first_p.remove(r)
    else:
        r_el = etree.SubElement(first_p, qn('a:r'))
        t_el = etree.SubElement(r_el, qn('a:t'))
        t_el.text = new_text


def set_cell(cell, text, bold=False, font_size=11):
    """重寫表格儲存格（清空再填）"""
    txBody = cell.text_frame._txBody
    for p in txBody.findall(qn('a:p')):
        txBody.remove(p)
    p = etree.SubElement(txBody, qn('a:p'))
    r = etree.SubElement(p, qn('a:r'))
    rPr = etree.SubElement(r, qn('a:rPr'),
                           attrib={'lang': 'zh-TW', 'dirty': '0',
                                   'b': '1' if bold else '0',
                                   'sz': str(int(font_size * 100))})
    t = etree.SubElement(r, qn('a:t'))
    t.text = text


def dup_slide_to_end(prs, src_idx):
    """複製 src_idx 的 slide XML，附加到最後，回傳新 slide"""
    src = prs.slides[src_idx]
    new = prs.slides.add_slide(prs.slide_layouts[6])   # Blank
    new_tree = new.shapes._spTree
    for child in list(new_tree):
        new_tree.remove(child)
    for child in src.shapes._spTree:
        new_tree.append(copy.deepcopy(child))
    return new


def move_slide(prs, old_idx, new_idx):
    """把第 old_idx 張 slide 移到 new_idx"""
    lst = prs.slides._sldIdLst
    items = list(lst)
    lst.remove(items[old_idx])
    lst.insert(new_idx, items[old_idx])


# ══════════════════════════════════════════════
# v8 → v9：只更新需要改的地方，保留使用者手動改好的 Slide 2、3
# INPUT = v8（已有 4 張：封面 / 方案比較 / 訓練規模 / 結論）
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Slide 1（封面）：情境改為 250hr，比較對象改為 Pro+
#   50hr GCP 實查 = $71.5 → 250hr = $71.5 × 5 = $357.5
#   Pro+ $49.99 vs GCP $357.5 ≈ 7.15× ≈ 約 7×
# ──────────────────────────────────────────────
slide1 = prs.slides[0]
for shape in slide1.shapes:
    if not shape.has_text_frame:
        continue
    n = shape.name
    if n == 'TextBox 5':
        set_first_run(shape, '250 小時 T4 訓練情境 ‧ 單位：美金 ‧ 2026/04/23 查價')
    elif n == 'TextBox 11':
        set_first_run(shape, 'Colab Pro+ (250 小時)')
    elif n == 'TextBox 12':
        set_first_run(shape, '$49.99')
    elif n == 'TextBox 13':
        set_first_run(shape, '月費內含 500 CU，跑 ~250hr T4 訓練')
    elif n == 'TextBox 16':
        set_first_run(shape, 'GCP on-demand (250 小時)')
    elif n == 'TextBox 17':
        set_first_run(shape, '$357.5')
    elif n == 'TextBox 23':
        set_first_run(shape, 'GCP 比 Colab Pro+ 貴 7 倍')

# ──────────────────────────────────────────────
# Slide 4（結論）：GCP 比較數字更新為 Pro+ 基準
#   TextBox 16：$71.5 vs $9.99 → $357.5 vs $49.99
# ──────────────────────────────────────────────
slide4 = prs.slides[3]   # v8 的 Slide 4（index=3）= 結論
for shape in slide4.shapes:
    if not shape.has_text_frame:
        continue
    n = shape.name
    if n == 'TextBox 16':
        set_first_run(shape,
            '‧ 成本貴 7×（$357.5 vs $49.99 / 250hr）；'
            '訓練完忘記關機 = 繼續計費，停機後磁碟仍持續計費；對單純訓練場景不划算')


# ══════════════════════════════════════════════
# 儲存
# ══════════════════════════════════════════════
prs.save(OUTPUT)
print(f'✅ 已儲存 → {OUTPUT}')
print(f'   總頁數：{len(prs.slides)} 張')
