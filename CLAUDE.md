# PPOCRv4 Mobile OCR 訓練專案

## 專案目的

使用 **PPOCRv4 mobile** 模型訓練自訂 OCR，最終目標是**部署至手機端（C++ ONNX Runtime）**。
應用場景：辨識產品上的日期（有效期限 EXP / 製造日期 MFG）。

訓練順序：**先訓練 det（文字偵測）模型**，完成後再訓練 rec（文字識別）模型。

---

## 完整 Pipeline

```
手機端 C++ (ONNX Runtime)
    ↓ 收集大量圖片
Python 環境標注 (PPOCRLabel)
    ↓ 產生 Label.txt
資料分割 (split_dataset.py)
    ↓ train/val
PaddleOCR 訓練 (train.py)
    ↓ best_accuracy.pdparams
匯出推論模型 (export_model.py)
    ↓ inference.pdmodel + .pdiparams
轉換 ONNX (convert_to_onnx.py)
    ↓ model.onnx
部署至 C++ / 手機端
```

---

## 目錄結構

```
claudeProject/
├── .claude/
│   └── settings.json              ← Claude 專案層級權限設定
├── dataset/
│   ├── images/                    ← 原始訓練圖片放這裡（標注前）
│   ├── det/
│   │   ├── train/                 ← det 訓練集圖片
│   │   ├── val/                   ← det 驗證集圖片
│   │   ├── train_label.txt        ← det 訓練標注
│   │   └── val_label.txt          ← det 驗證標注
│   └── rec/                       ← （未來）rec 訓練資料
│       ├── train/
│       ├── val/
│       ├── train_label.txt
│       └── val_label.txt
├── configs/
│   ├── det/
│   │   └── PP-OCRv4_mobile_det_finetune.yml
│   └── rec/
│       └── PP-OCRv4_mobile_rec_finetune.yml
├── pretrained_models/             ← 預訓練權重
├── output/
│   ├── det/                       ← det 訓練產出
│   │   ├── best_accuracy.*        ← 最佳模型
│   │   ├── inference/             ← 匯出的推論模型
│   │   └── onnx/                  ← 轉換後的 ONNX
│   └── rec/                       ← rec 訓練產出
├── eval_cpp_runner/               ← C++ ONNX 推論測試工具
│   ├── models/                    ← ONNX 模型檔（部署測試用）
│   └── ocr_cpp/                   ← C++ 推論原始碼
└── tools/
    ├── split_dataset.py           ← 資料集分割
    ├── export_model.py            ← 匯出推論模型
    ├── convert_to_onnx.py         ← 轉換為 ONNX
    └── download_pretrained.py     ← 下載預訓練模型
```

---

## 工具路徑

| 工具 | 路徑 |
|------|------|
| PaddleOCR 原始碼 | `C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR` |
| PaddleOCR 訓練腳本 | `C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR\tools\train.py` |
| PPOCRLabel 標記工具 | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel` |
| PPOCRLabel venv Python | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel\venv\Scripts\python.exe` |
| C++ 推論測試 | `eval_cpp_runner/build/Release/eval_runner.exe` |

---

## 標注圖片（PPOCRLabel）

啟動指令：
```cmd
cd C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel
venv\Scripts\python.exe PPOCRLabel.py --lang ch
```

標注完成後，PPOCRLabel 會在圖片目錄下產生：
- `Label.txt`：標注資料（格式：`圖片路徑\t[{...bbox...}]`）
- `fileState.txt`：標注狀態

---

## Det 模型訓練流程（目前重點）

### 1. 下載預訓練模型
```cmd
python tools/download_pretrained.py --model det
python tools/download_pretrained.py --model det_train
```

### 2. 準備資料
1. 將原始圖片放入 `dataset/images/`
2. 用 PPOCRLabel 標注 → 產生 `Label.txt`
3. 執行分割腳本：
   ```cmd
   python tools/split_dataset.py --label_path dataset/Label.txt --output_dir dataset/det --copy_images
   ```

### 3. 開始訓練
```cmd
cd C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR
python tools/train.py -c C:\Users\andy_ac_chen\Desktop\claudeProject\configs\det\PP-OCRv4_mobile_det_finetune.yml
```

### 4. 匯出推論模型
```cmd
python tools/export_model.py --model_type det
```

### 5. 轉換為 ONNX
```cmd
python tools/convert_to_onnx.py --model_type det
```

### 6. 部署測試
將 `output/det/onnx/ch_PP-OCRv4_det_infer.onnx` 複製到 `eval_cpp_runner/models/` 進行測試。

---

## Rec 模型訓練流程（未來）

### 1. 下載預訓練模型
```cmd
python tools/download_pretrained.py --model rec
```

### 2. 準備 Rec 訓練資料
Rec 標注格式（`train_label.txt`）：
```
圖片路徑\t辨識文字
crop_img_001.jpg	2025/03/15
crop_img_002.jpg	EXP:2026.01
```
可用 det 模型裁切出文字區域後手動標注，或用 PPOCRLabel 的 rec 模式。

### 3. 開始訓練
```cmd
cd C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR
python tools/train.py -c C:\Users\andy_ac_chen\Desktop\claudeProject\configs\rec\PP-OCRv4_mobile_rec_finetune.yml
```

### 4. 匯出 + 轉 ONNX
```cmd
python tools/export_model.py --model_type rec
python tools/convert_to_onnx.py --model_type rec
```

---

## C++ 部署說明

`eval_cpp_runner/` 使用 ONNX Runtime 進行推論：
- 模型格式：`.onnx`
- 推論框架：ONNX Runtime 1.23.2
- 模型檔案放在 `eval_cpp_runner/models/` 下：
  - `ch_PP-OCRv4_det_infer.onnx`（偵測）
  - `ch_PP-OCRv4_rec_infer.onnx`（辨識）
  - `ch_ppocr_mobile_v2.0_cls_infer.onnx`（分類器）
  - `ppocr_keys_v1.txt`（字元字典）

---

## Claude 使用注意事項

- 大多數 Bash 指令已設定為自動允許（見 `.claude/settings.json`）
- 以下操作仍需確認或已被拒絕：
  - `git push --force` / `-f`（防止強制推送）
  - `rm -rf /` 或 `rm -rf C:` 等大範圍刪除
  - `format`、`shutdown`、`reg delete` 等系統危險指令
