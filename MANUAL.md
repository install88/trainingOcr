# 操作手冊 — PPOCRv4 Mobile OCR 訓練專案

> 這份手冊記錄所有日常操作的完整指令，不需要記命令，照著做就可以。

---

## 目錄

1. [新增標注圖片後的整理流程](#1-新增標注圖片後的整理流程)
2. [自動標注新圖片（auto_label）](#2-自動標注新圖片auto_label)
3. [準備 Colab 訓練資料（打包上傳）](#3-準備-colab-訓練資料打包上傳)
4. [Colab 訓練 det 模型（完整步驟）](#4-colab-訓練-det-模型完整步驟)
5. [下載訓練結果並部署](#5-下載訓練結果並部署)
6. [在另一台電腦安裝標注環境](#6-在另一台電腦安裝標注環境)
7. [常用指令速查](#7-常用指令速查)

---

## 1. 新增標注圖片後的整理流程

每次你用 PPOCRLabel 標完新圖片之後，執行以下 3 個步驟讓資料生效。

**前提：** 圖片和 Label.txt 都在 `C:\Users\andy_ac_chen\success\日期\` 或 `fail\日期\` 裡面。

### Step 1：重新匯入所有標注（import）

```cmd
cd C:\Users\andy_ac_chen\Desktop\claudeProject

python tools/import_labeled_data.py ^
  --source C:\Users\andy_ac_chen\success --prefix s ^
  --source C:\Users\andy_ac_chen\fail    --prefix f ^
  --output dataset/images
```

完成後會顯示每個日期資料夾有幾筆 ok，以及 `Total merged label lines: XXX`。

### Step 2：重新分割 train / val

```cmd
rmdir /s /q dataset\det\train dataset\det\val

python tools/split_dataset.py ^
  --label_path dataset/images/Label.txt ^
  --output_dir dataset/det ^
  --copy_images ^
  --ratio 0.8
```

完成後會顯示 `訓練集: XXX 筆 | 驗證集: XXX 筆`。

### Step 3：重新打包 zip（給 Colab 用）

```cmd
cd dataset\det
python -c "
import zipfile, os
out = '../det_dataset.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk('.'):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, '.'))
print('完成，大小:', round(os.path.getsize(out)/1024/1024, 2), 'MB')
"
cd ..
```

完成後 `dataset/det_dataset.zip` 就是要上傳到 Google Drive 的檔案。

---

## 2. 自動標注新圖片（auto_label）

如果你有一整批新圖片想快速產生 Label.txt 草稿，不用一張一張手動框，用這個工具。

### 使用方式

```cmd
cd C:\Users\andy_ac_chen\Desktop\claudeProject

"C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel\venv\Scripts\python.exe" ^
  tools/auto_label.py ^
  --img_dir C:\Users\andy_ac_chen\success\20260419
```

把 `20260419` 換成你的日期資料夾名稱。

### 輸出結果說明

```
✓ date found  : 73   ← 自動找到日期，Label.txt 已寫好
⚠ FALLBACK    : 8    ← 找不到日期，用了備用框，difficult=true（紅框）
✗ skipped     : 1    ← 完全沒偵測到文字（模糊/空白）
```

### 跑完之後

1. 開 PPOCRLabel，選 `C:\Users\andy_ac_chen\success` 資料夾
2. 進到該日期資料夾，找**紅框**（`difficult=true`）的圖片
3. 手動修正框的位置和文字內容
4. 按 **Save**，Label.txt 自動更新
5. 回到 [Step 1](#step-1重新匯入所有標注import) 重新匯入

### 標注規則重點

| 類型 | 要標 | 範例 |
|------|------|------|
| EXP / BB / 有效日期（整段） | ✅ | `EXP:2026.10.18` |
| MFG / MFD / 製造日期（整段） | ✅ | `MFG 2026/03/10` |
| 有效期間 / 保存期限 | ✅ | `有效期間:3年` |
| 提示字在上、日期在下 | 只標日期那行 | 下方 `2026.07.26` |
| 日期後面有批號/附加碼 | 只保留日期 | `20260328`（去掉 AM20）|
| 地址、電話、成分、批號 | ❌ 不標 | |
| 時間戳記（13:04、01:58）| ❌ 不標 | |

---

## 3. 準備 Colab 訓練資料（打包上傳）

### 你需要上傳的檔案

| 檔案 | 上傳位置（Google Drive） |
|------|-------------------------|
| `dataset/det_dataset.zip` | `我的雲端硬碟/ocr_project/det_dataset.zip` |

> ⚠️ 路徑一定要完全符合，Colab notebook 寫死讀這個路徑。

### 上傳步驟

1. 打開 Google Drive（drive.google.com）
2. 在 `我的雲端硬碟` 下建立資料夾 `ocr_project`（如果沒有的話）
3. 進入 `ocr_project` 資料夾
4. 上傳 `C:\Users\andy_ac_chen\Desktop\claudeProject\dataset\det_dataset.zip`
5. 如果之前已有舊的 zip，直接覆蓋上傳

---

## 4. Colab 訓練 det 模型（完整步驟）

### 4.1 開啟 Notebook

1. 打開 https://colab.research.google.com/
2. **File → Open notebook → GitHub**
3. 輸入 `install88/trainingOcr`，按 Enter
4. 選擇 `notebooks/train_det_colab.ipynb`
5. **Runtime → Change runtime type → T4 GPU → Save**

### 4.2 依序執行每個 Cell

| Cell | 內容 | 預計時間 | 確認重點 |
|------|------|---------|---------|
| Cell 2 | `!nvidia-smi` | 2 秒 | 看到 `Tesla T4` |
| Cell 4 | Clone repo + 安裝套件 | 3~5 分鐘 | 最後沒有紅色 Error |
| Cell 5 | 驗證 PaddlePaddle | 5 秒 | `GPU available: True` |
| Cell 6 | Clone PaddleOCR | 30 秒 | 顯示 `done` |
| Cell 7 | Mount Drive + 解壓資料集 | 30~60 秒 | 需要授權 Google 帳號；看到 `train/: 744 files` |
| Cell 9 | 檢查資料集結構 | 5 秒 | 看到 train 和 val 有圖片 |
| Cell 11 | 下載預訓練模型 | 30 秒 | 看到 `best_accuracy.pdparams` |
| Cell 13 | 設定 config 路徑 | 1 秒 | 印出 config 前幾行 |
| **Cell 15** | ⭐ **Baseline 評估** | 1~2 分鐘 | **截圖！記下 hmean / precision / recall** |
| Cell 16 | （選用）從 checkpoint 恢復 | — | 正常不用跑 |
| **Cell 17** | ⭐ **訓練 200 epochs** | 1~2 小時 | loss 數字要一路往下掉 |
| **Cell 18** | ⭐ **訓練後評估** | 1~2 分鐘 | **截圖！和 Cell 15 對比 hmean** |
| Cell 20 | 匯出推論模型 | 20 秒 | 產生 `inference.pdmodel` |
| Cell 22 | 轉 ONNX | 30 秒 | 產生 `.onnx` 檔 |
| Cell 23 | 驗證 ONNX | 5 秒 | 印出 input/output shape |
| Cell 25 | 備份到 Drive | 10 秒 | 儲存到 `MyDrive/output_det/` |
| Cell 26 | **下載 ONNX** | 10 秒 | 瀏覽器自動下載 `.onnx` |

### 4.3 訓練效果判讀

訓練完後比較 Cell 15 和 Cell 18 的數字：

```
Cell 15 (Baseline，官方模型，沒在你資料上訓練):
  hmean: 0.XX

Cell 18 (After，fine-tune 完):
  hmean: 0.XX  ← 應該比 baseline 高
```

| hmean 進步幅度 | 判讀 |
|--------------|------|
| > 0.05 | ✅ fine-tune 有效 |
| 0.02 ~ 0.05 | ⚠️ 有效但幅度小，可考慮加更多資料 |
| < 0.02 | ⚠️ 效果不明顯，資料可能太少 |
| 反而下降 | ❌ Overfit，可試試降低 epoch 或 learning_rate |

### 4.4 注意事項

- Colab 免費版閒置 90 分鐘會斷線，訓練中**不要關瀏覽器 tab**
- 如果斷線後重連，用 **Cell 16**（從 checkpoint 恢復）繼續訓練
- 所有訓練產出自動備份在 `MyDrive/output_det/`

---

## 5. 下載訓練結果並部署

### 取得 ONNX 模型

Cell 26 執行後，瀏覽器會自動下載 `ch_PP-OCRv4_det_infer.onnx`。

或者從 Google Drive 下載：`MyDrive/output_det/onnx/ch_PP-OCRv4_det_infer.onnx`

### 放到 C++ 測試環境

```
下載的 .onnx 檔 → 複製到：
C:\Users\andy_ac_chen\Desktop\claudeProject\eval_cpp_runner\models\ch_PP-OCRv4_det_infer.onnx
```

覆蓋舊的檔案，用 C++ eval runner 測試效果。

### 本機 Python 推論測試（畫紅框）

```cmd
cd C:\Users\andy_ac_chen\Desktop\claudeProject

python tools/run_ocr.py ^
  --img_dir dataset/det/val ^
  --out_dir output/det/vis
```

輸出的圖片會在 `output/det/vis/`，每張圖有紅色 bounding box 和 JSON 結果。

---

## 6. 在另一台電腦安裝標注環境

### 安裝步驟

```cmd
git clone https://github.com/PaddleOCR-Community/PPOCRLabel.git
cd PPOCRLabel
python -m venv venv
venv\Scripts\activate
pip install paddlepaddle==2.6.2 paddleocr==2.9.1 pyqt5==5.15.11 xlrd==2.0.2
python PPOCRLabel.py --lang ch
```

### 需求

- **Python 3.11.x**（建議，3.12 也可以）
- Windows 10/11
- 網路連線（安裝套件時需要，約下載 2~3 GB）

### 取得本專案的工具腳本

```cmd
git clone https://github.com/install88/trainingOcr.git
```

---

## 7. 常用指令速查

```cmd
:: ── 進入專案目錄 ──────────────────────────────────────
cd C:\Users\andy_ac_chen\Desktop\claudeProject

:: ── 匯入所有標注（每次加新資料後都要跑）────────────────
python tools/import_labeled_data.py ^
  --source C:\Users\andy_ac_chen\success --prefix s ^
  --source C:\Users\andy_ac_chen\fail    --prefix f ^
  --output dataset/images

:: ── 重新分割 train/val ──────────────────────────────
rmdir /s /q dataset\det\train dataset\det\val
python tools/split_dataset.py ^
  --label_path dataset/images/Label.txt ^
  --output_dir dataset/det ^
  --copy_images ^
  --ratio 0.8

:: ── 查看目前有幾筆標注 ──────────────────────────────
python -c "
with open('dataset/images/Label.txt', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]
print(f'dataset/images/Label.txt: {len(lines)} 筆')
"

:: ── 自動標注新圖片（產生草稿）────────────────────────
"C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel\venv\Scripts\python.exe" ^
  tools/auto_label.py ^
  --img_dir C:\Users\andy_ac_chen\success\20260419

:: ── 開啟 PPOCRLabel ──────────────────────────────────
cd C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel
venv\Scripts\activate
python PPOCRLabel.py --lang ch

:: ── 打包 zip（完整版） ──────────────────────────────
cd C:\Users\andy_ac_chen\Desktop\claudeProject\dataset\det
python -c "
import zipfile, os
out = '../det_dataset.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk('.'):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, '.'))
print('完成:', round(os.path.getsize(out)/1024/1024, 2), 'MB')
"

:: ── git push（需在本機 CMD 手動執行）──────────────────
cd C:\Users\andy_ac_chen\Desktop\claudeProject
git add -A
git commit -m "update dataset"
git push
```

---

## 附錄：資料夾標注狀況（2026-04-20 更新）

| 來源 | 日期 | 圖片數 | 標注狀態 |
|------|------|--------|---------|
| success | 20260325~20260414 | 533 | ✅ 已標注 |
| success | 20260417 | 47 | ✅ 已標注 |
| success | 20260418 | 69 | ✅ 已標注（66 筆） |
| success | 20260419 | 82 | ✅ auto_label 草稿（73✓ 8⚠ 1✗）需 PPOCRLabel 確認紅框 |
| fail | 20260325~20260419 | 219 | ✅ 已標注（203 筆） |
| **合計** | | **952 張** | **930 筆已匯入訓練集** |

train: **744 張** / val: **186 張**
