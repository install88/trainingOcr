# PPOCRv4 Mobile OCR — 完整工作流程文件

> **最後更新：2026-03-31**
> **GitHub Repo：** https://github.com/install88/trainingOcr

---

## 目標

在手機端（C++ ONNX Runtime）辨識產品上的**日期文字**（EXP 有效期限 / MFG 製造日期），
透過 fine-tune PPOCRv4 mobile 模型提升偵測與辨識準確率。

---

## 整體架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                        手機端 (C++)                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ 拍照/取圖 │──▶│ Det 偵測  │──▶│ Rec 辨識  │──▶│ 日期後處理   │  │
│  └──────────┘   │ (ONNX)   │   │ (ONNX)   │   │ EXP/MFG解析 │  │
│                  └──────────┘   └──────────┘   └─────────────┘  │
│                  ONNX Runtime 1.23.2                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ 收集大量圖片
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Python 訓練環境                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ PPOCRLabel│──▶│ 資料分割  │──▶│ PaddleOCR │──▶│ 匯出+轉ONNX │  │
│  │ 標注工具  │   │ 80/20    │   │ 訓練      │   │             │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘  │
│                                                       │          │
└───────────────────────────────────────────────────────┼──────────┘
                                                        │
                                              .onnx 模型回到手機端
```

---

## 訓練順序

```
Phase 1（目前）：訓練 Det 模型 → 讓紅框精準框到日期區域
Phase 2（未來）：訓練 Rec 模型 → 提升日期文字辨識正確率
```

---

## 完整流程（Step by Step）

### Phase 0：環境準備（已完成）

| 步驟 | 說明 | 狀態 |
|------|------|------|
| 本機 Python 環境 | Python 3.11 + PaddleOCR 3.4.0 + PaddlePaddle 3.0.0 (CPU) | ✅ |
| PPOCRLabel 標注工具 | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel` | ✅ |
| PaddleOCR 原始碼 | `C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR` | ✅ |
| 專案 repo | https://github.com/install88/trainingOcr | ✅ |
| 本機推論腳本 | `tools/run_ocr.py`（紅框圖 + JSON + C++比較） | ✅ |
| Colab 訓練 notebook | `notebooks/train_det_colab.ipynb` / `train_rec_colab.ipynb` | ✅ |

---

### Phase 1：Det 模型訓練

#### Step 1 — 收集圖片

- 從手機端收集產品日期照片
- 放入 `dataset/images/`
- **建議數量：先 50 張驗證流程，目標 200~500 張**

#### Step 2 — PPOCRLabel 標注

```cmd
cd C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel
venv\Scripts\python.exe PPOCRLabel.py --lang ch
```

**操作流程：**
1. 檔案 → 開啟資料夾 → 選擇 `dataset/images/`
2. 用矩形框工具框選**日期文字區域**
3. 輸入 transcription（框內文字，如 `2026.05.10`）
4. 儲存（Ctrl+S）
5. 標注完所有圖片後，`dataset/images/` 下會產生 `Label.txt`

**標注重點：**
```
✅ 只框日期相關文字：
   "2026.05.10"、"EXP:2027.10.08"、"有效2026.01.20"、"20260409"

❌ 不要框無關文字：
   品名、成分表、條碼、logo
```

**Label.txt 格式（PPOCRLabel 自動產生）：**
```
images/photo_001.jpg	[{"transcription": "2026.05.10", "points": [[100,50],[250,50],[250,80],[100,80]], "difficult": false}]
images/photo_002.jpg	[{"transcription": "EXP:2027.10.08", "points": [[30,120],[280,120],[280,160],[30,160]], "difficult": false}]
```

#### Step 3 — 分割資料集

```cmd
cd C:\Users\andy_ac_chen\Desktop\claudeProject

python tools/split_dataset.py ^
    --label_path dataset/images/Label.txt ^
    --output_dir dataset/det ^
    --copy_images ^
    --ratio 0.8
```

**產出：**
```
dataset/det/
├── train/              ← 80% 圖片
├── val/                ← 20% 圖片
├── train_label.txt     ← 訓練標注
└── val_label.txt       ← 驗證標注
```

#### Step 4 — 打包上傳

```cmd
:: 在 dataset/det/ 目錄下壓縮成 zip
:: 上傳到 Google Drive 的 My Drive/ocr_project/det_dataset.zip
```

**zip 內容：**
```
det_dataset.zip
├── train/
├── val/
├── train_label.txt
└── val_label.txt
```

#### Step 5 — Colab 訓練

1. 開啟 Google Colab（選 GPU runtime：T4）
2. 上傳或開啟 `notebooks/train_det_colab.ipynb`
3. 按順序執行每個 cell：

```
Cell 0  → 確認 GPU
Cell 1  → git clone repo + 安裝套件
Cell 2  → 驗證 PaddlePaddle
Cell 3  → 掛載 Google Drive + 解壓資料集
Cell 4  → 檢查資料集
Cell 5  → 下載預訓練模型
Cell 6  → 確認 config
Cell 7  → 開始訓練 ← 主要等待時間
Cell 8  → 評估模型（看 hmean 指標）
Cell 9  → 匯出推論模型
Cell 10 → 轉 ONNX
Cell 11 → 驗證 ONNX
Cell 12 → 下載模型
```

**訓練時間預估（Colab T4 GPU）：**
| 資料量 | Epoch 200 預估時間 |
|--------|-------------------|
| 50 張  | ~30 分鐘 |
| 200 張 | ~1~2 小時 |
| 500 張 | ~3~5 小時 |

#### Step 6 — 部署測試

1. 從 Colab 下載 `ch_PP-OCRv4_det_infer.onnx`
2. 複製到 `eval_cpp_runner/models/` 覆蓋原本的 det 模型
3. 執行 C++ 測試：
   ```cmd
   eval_cpp_runner\build\Release\eval_runner.exe eval_cpp_runner\input\success\
   ```
4. 或用 Python 測試：
   ```cmd
   python tools/run_ocr.py --input eval_cpp_runner/input/success/ --det_model_dir output/det/inference/
   ```

---

### Phase 2：Rec 模型訓練（未來）

#### Step 1 — 準備 Rec 訓練資料

用已訓練好的 det 模型裁切出文字區域，再手動標注辨識文字。

**Rec 標注格式（`train_label.txt`）：**
```
crop_001.jpg	2025/03/15
crop_002.jpg	EXP:2026.01
crop_003.jpg	有效2026.01.20
crop_004.jpg	20260409
```

#### Step 2 — 打包上傳

```
rec_dataset.zip
├── train/              ← 裁切後的文字圖片
├── val/
├── train_label.txt
└── val_label.txt
```

#### Step 3 — Colab 訓練

開啟 `notebooks/train_rec_colab.ipynb`，流程同 det。

#### Step 4 — 部署

下載 `ch_PP-OCRv4_rec_infer.onnx` → 放入 `eval_cpp_runner/models/`

---

## 專案目錄結構

```
trainingOcr/                        ← GitHub Repo
│
├── .claude/settings.json            ← Claude 權限設定
├── .gitignore
├── CLAUDE.md                        ← Claude 專案說明
├── WORKFLOW.md                      ← 本文件
├── requirements.txt                 ← 雲端 GPU server 用
├── requirements_local.txt           ← 本機 CPU 用
│
├── configs/
│   ├── det/
│   │   └── PP-OCRv4_mobile_det_finetune.yml
│   └── rec/
│       └── PP-OCRv4_mobile_rec_finetune.yml
│
├── tools/
│   ├── run_ocr.py                   ← 本機推論（紅框圖 + JSON）
│   ├── split_dataset.py             ← 資料集分割 (80/20)
│   ├── export_model.py              ← 匯出 Paddle inference 模型
│   ├── convert_to_onnx.py           ← 轉 ONNX
│   └── download_pretrained.py       ← 下載預訓練模型
│
├── notebooks/
│   ├── train_det_colab.ipynb        ← Colab det 訓練
│   └── train_rec_colab.ipynb        ← Colab rec 訓練
│
├── dataset/                          ← 圖片不進 git（.gitignore）
│   ├── images/                       ← 原始圖片
│   ├── det/{train,val}/              ← 分割後 det 資料
│   └── rec/{train,val}/              ← 分割後 rec 資料
│
├── pretrained_models/                ← 預訓練權重（不進 git）
├── output/                           ← 訓練產出（不進 git）
│
└── eval_cpp_runner/                  ← C++ ONNX 推論（不進 git）
    ├── models/                       ← .onnx 模型檔
    ├── ocr_cpp/                      ← C++ 原始碼
    ├── input/                        ← 測試圖片
    └── out/                          ← 推論結果
```

---

## 常用指令速查

### 本機操作

```cmd
:: 啟動 PPOCRLabel 標注
cd C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel
venv\Scripts\python.exe PPOCRLabel.py --lang ch

:: 分割資料集
cd C:\Users\andy_ac_chen\Desktop\claudeProject
python tools/split_dataset.py --label_path dataset/images/Label.txt --output_dir dataset/det --copy_images

:: 本機 OCR 推論（產生紅框圖 + JSON）
python tools/run_ocr.py --input eval_cpp_runner/input/success/ --output_dir output/ocr_results/

:: 本機推論 + 與 C++ 結果比較
python tools/run_ocr.py --input eval_cpp_runner/input/success/ --compare eval_cpp_runner/out/eval_results.jsonl

:: 只跑偵測（檢查紅框效果）
python tools/run_ocr.py --input test.jpg --det_only

:: 下載預訓練模型
python tools/download_pretrained.py --model det
python tools/download_pretrained.py --model rec

:: 匯出 + 轉 ONNX（本機訓練完後）
python tools/export_model.py --model_type det
python tools/convert_to_onnx.py --model_type det
```

### Colab 操作

```python
# Clone repo
!git clone https://github.com/install88/trainingOcr.git

# Clone PaddleOCR
!git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git

# 安裝（CUDA 12.x）
!pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 訓練 det
!python tools/train.py -c /content/trainingOcr/configs/det/PP-OCRv4_mobile_det_finetune.yml \
    -o Global.use_gpu=true ...

# 訓練 rec
!python tools/train.py -c /content/trainingOcr/configs/rec/PP-OCRv4_mobile_rec_finetune.yml \
    -o Global.use_gpu=true ...
```

### Git 操作

```cmd
cd C:\Users\andy_ac_chen\Desktop\claudeProject
git add -A
git commit -m "描述"
git push
```

---

## 模型檔案對照

| 用途 | 檔名 | 來源 |
|------|------|------|
| Det 偵測 | `ch_PP-OCRv4_det_infer.onnx` | 訓練後匯出 |
| Rec 辨識 | `ch_PP-OCRv4_rec_infer.onnx` | 訓練後匯出 |
| Cls 分類器 | `ch_ppocr_mobile_v2.0_cls_infer.onnx` | 官方預訓練（不需重新訓練） |
| 字元字典 | `ppocr_keys_v1.txt` | PaddleOCR 內建（6623 字元） |

---

## 評估指標

### Det 模型
- **hmean**（F1-score）：precision 和 recall 的調和平均
- 目標：hmean > 0.8

### Rec 模型
- **acc**（accuracy）：字元級正確率
- 目標：acc > 0.9

---

## 疑難排解

### Colab 斷線後恢復訓練
```python
!python tools/train.py \
    -c /content/trainingOcr/configs/det/PP-OCRv4_mobile_det_finetune.yml \
    -o Global.checkpoints=/content/output/det/latest \
       ...其他路徑覆寫...
```

### 訓練 Loss 不下降
- 檢查標注是否正確（用 PPOCRLabel 重新打開確認）
- 降低 learning_rate（config 裡改 `learning_rate: 0.0001`）
- 增加資料量

### ONNX 轉換失敗
- 確認 paddle2onnx 版本 >= 1.3.0
- 嘗試更高的 opset_version：`--opset_version 14`

### 本機 run_ocr.py 很慢
- 這是 CPU 推論，每張 ~3 秒是正常的
- 首次執行會下載模型，之後會快一些
