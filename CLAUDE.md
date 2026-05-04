# PPOCRv5 Mobile OCR 訓練專案

## 專案目的

辨識產品包裝上的日期（有效期限 EXP / 製造日期 MFG / 保存期限），最終部署至**手機端 C++ ONNX Runtime**。

---

## 目前進度（2026-05-04 更新，下午）

| 階段 | 狀態 | 備註 |
|------|------|------|
| 專案架構 | ✅ | configs / tools / notebooks 就位 |
| 標注環境 | ✅ | PPOCRLabel，見 MANUAL.md |
| 資料集（det） | ✅ | **1,828 張**（train 1462 / val 366），backup 共 1,829 jpg |
| det dataset zip | ✅ | det_dataset.zip（84.9 MB），2026-04-30 打包，在 Google Drive |
| det v5 訓練（第一輪） | ✅ | hmean **89.9%**，best_epoch=25，已轉 ONNX |
| det v5 訓練（第二輪） | 🔜 **待執行** | config 已改好，需 push 後重跑 Colab |
| 資料集（rec） | ✅ | **~1,792 train / ~512 val**（crop 比舊版翻倍） |
| rec dataset zip | ✅ | rec_dataset.zip，在 Google Drive |
| rec 訓練 | ✅ | best acc **91.86%**（best_epoch≈89），已完成 |
| rec 轉 ONNX | 🔜 **待執行** | 訓練完成，尚未匯出 |
| compare 工具驗證 | 🔜 **待執行** | 等 det 第二輪 ONNX 出來後跑 |
| C++ 部署測試 | 🔜 未開始 | |
| Colab Pro+ 成本簡報 | ✅ | `colab_vs_gcp成本_v2.pptx`，5 張，附訓練規模數字 |

---

## ⚠️ 當前最重要的待辦（依序執行）

1. **git push**（本機 CMD 手動執行，Claude 無 TTY）
   ```cmd
   cd C:\Users\andy_ac_chen\Desktop\claudeProject
   git push
   ```
   - `eb4320c`：det config 修正（shrink_ratio/unclip/Fliplr/Resize）← **已 commit，尚未 push**
   - `e3f0175`：compare thresh 對齊 + CLAUDE.md 更新 ← **已 commit，尚未 push**

2. **Colab 重跑 det 訓練**（`train_det_v5_colab.ipynb`，**從 pretrained 開始，不是 epoch 25**）
   - 原因：shrink_ratio 從 0.4→0.3，前一輪 checkpoint 學到的 prob map 不相容，接續訓練會混亂

3. **det best_model → 轉 ONNX** → 下載到 `eval_cpp_runner/models/ch_PP-OCRv4_det_infer_new.onnx`

4. **rec best_accuracy → 轉 ONNX** → 下載到 `eval_cpp_runner/models/ch_PP-OCRv4_rec_infer_new.onnx`

5. **跑 compare 驗證**：`python tools\compare_det_onnx.py --date 20260428 --n 15`
   - 重點驗證：① 框是否不再截短（Det 修正效果）② 字母不再被換成數字（Rec 改善效果）

---

## 已知問題與解決方案

### Det 框切短問題（已在 config 修正，等重訓）
- **症狀**：`有效日期:2027.10.02` 框只偵測到 `27.10.02`（`有效日期:20` 被切掉）
- **根本原因**：`shrink_ratio=0.4` 太激進，訓練時把前綴像素縮到 ground truth 之外，probability map 沒有前綴訊號
- **推論參數（thresh、unclip_ratio）已確認無效**，一定要重訓才能修
- **Config 修正內容**（已改，等 push）：

| 參數 | 舊值 | 新值 |
|------|------|------|
| MakeBorderMap shrink_ratio | 0.4 | **0.3** |
| MakeShrinkMap shrink_ratio | 0.4 | **0.3** |
| PostProcess unclip_ratio | 1.5 | **2.0** |
| IaaAugment Fliplr | p=0.5 | **移除** |
| Resize size | [0.5, 3] | **[0.5, 2]** |

### Rec 字母→數字替換問題（Catastrophic Forgetting）
- **症狀**：`E`→`2`、`N`→`1`（舊版 rec ONNX）
- **原因**：fine-tune 資料以數字日期為主，字母特徵被覆蓋
- **新版 rec（91.86% acc）是否改善**：待匯出 ONNX 後用 compare 工具驗證

---

## 模型版本對照

| 檔案 | 內容 | 來源 |
|------|------|------|
| `eval_cpp_runner/models/ch_PP-OCRv4_det_infer.onnx` | Det pretrained（原廠） | PaddleOCR 官方 |
| `eval_cpp_runner/models/ch_PP-OCRv4_det_infer_new.onnx` | Det finetuned v5（第一輪，hmean 89.9%） | output_det_v5/best_model epoch 25 |
| `eval_cpp_runner/models/ch_PP-OCRv4_rec_infer.onnx` | Rec pretrained（原廠） | PaddleOCR 官方 |
| `eval_cpp_runner/models/ch_PP-OCRv4_rec_infer_new.onnx` | Rec finetuned（acc 91.86%，**待更新**） | output_rec/best_accuracy |

---

## Compare 工具參數（tools/compare_det_onnx.py）

- **thresh=0.3**（對齊 config PostProcess）
- **box_thresh=0.4**（原廠 0.6 → 降低救低信心日期框）
- **unclip_ratio=2.0**（對齊 config PostProcess）
- **use_dilation=True**（橋接斷點文字）

執行範例：
```powershell
cd C:\Users\andy_ac_chen\Desktop\claudeProject
python tools\compare_det_onnx.py --date 20260428 --n 15
# 指定特定資料夾：
python tools\compare_det_onnx.py --success C:\path\s --fail C:\path\f --n 20
```

---

## 資料集現況（2026-05-04）

### Det dataset
- **Labeled**: train 1,462 / val 366（總 1,828 張）
- **Zip**: `dataset/det_dataset.zip`（84.9 MB，含 train 1558 + val 462 張，多的 96 張無 label 無害）
- **來源**: `C:\Users\andy_ac_chen\Desktop\backup\success\` + `backup\fail\`
- **Label 品質**: 良好，弧形/截斷邊緣等特殊案例有合理標記

### Rec dataset
- **Labeled**: train ~1,792 / val ~512（比舊版 833/208 翻倍）
- **Zip**: `dataset/rec_dataset.zip`，在 Google Drive

---

## 原始資料來源

```
C:\Users\andy_ac_chen\Desktop\backup\success\<日期>\  ← 標注完成
C:\Users\andy_ac_chen\Desktop\backup\fail\<日期>\    ← 標注完成
```
（注意：原本在 `C:\Users\andy_ac_chen\success\` 和 `fail\`，現在 backup 是主要位置）

每個日期資料夾內：`*.jpg`（600×373）、`Label.txt`（PPOCRLabel 格式）、`fileState.txt`

---

## 標注規則（核心）

- 只標日期相關資訊（EXP / MFG / MFD / 有效期間 / 製造日期 / 保存期限）
- 不標：地址、電話、成分、批號、時間碼、一般文字
- **視覺上同一列 = 一個框**（不拆分、不合并跨列）
- 前綴有業務意義時保留（`EXP:2026.01.01` → 整段標）
- 提示字在上、日期在下 → 只標日期那行
- 日期後有批碼且同行 → 可整行標（`2025.10.03 AJ08` 可寫完整）
- 不確定的圖片（看不清、模糊）→ 直接丟掉，不強制標

---

## 工具路徑

| 工具 | 路徑 |
|------|------|
| 本專案 | `C:\Users\andy_ac_chen\Desktop\claudeProject` |
| PaddleOCR | `C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR` |
| PPOCRLabel | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel` |
| PPOCRLabel Python | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel\venv\Scripts\python.exe` |
| GitHub Repo | https://github.com/install88/trainingOcr |
| Google Drive | `ocr_project/` → `output_det_v5/`、`output_rec/`、zip 檔案 |

---

## Claude 使用注意事項

- Bash 指令大多自動允許（見 `.claude/settings.json`）
- git push 需在本機 CMD 手動執行（Claude 無 TTY）
- 禁止：`git push --force`、大範圍 `rm -rf`、`format`、`shutdown`
