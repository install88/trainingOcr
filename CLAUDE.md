# PPOCRv4 Mobile OCR 訓練專案

## 專案目的

使用 **PPOCRv4 mobile** 模型訓練自訂 OCR，最終目標是**部署至手機端（C++ ONNX Runtime）**。
應用場景：辨識產品上的日期（有效期限 EXP / 製造日期 MFG）。

訓練順序：**先訓練 det（文字偵測）模型**，完成後再訓練 rec（文字識別）模型。

---

## 目前進度（給 Claude 快速上手用）

| 階段 | 狀態 | 備註 |
|------|------|------|
| 專案架構建立 | ✅ 完成 | configs / tools / notebooks 全部就位 |
| PPOCRLabel 標注環境 | ✅ 完成 | 另一台電腦也能安裝（見 MANUAL.md） |
| 資料匯入腳本 | ✅ 完成 | tools/import_labeled_data.py |
| 自動標注腳本 | ✅ 完成 | tools/auto_label.py（用 PaddleOCR 辨識後產生草稿） |
| 資料集分割 | ✅ 完成 | train 744 / val 186（共 930 筆，2026-04-20 更新） |
| det dataset zip | ✅ 完成 | dataset/det_dataset.zip（39.5 MB） |
| Colab det notebook | ✅ 完成 | notebooks/train_det_colab.ipynb（Drive checkpoint + resume 區塊 A/B） |
| det 模型訓練 | ✅ 完成 | Colab 跑到 epoch 191，best_accuracy 在 epoch 135（hmean 0.834） |
| rec dataset 生成 | ✅ 完成 | tools/prepare_rec_dataset.py → 833 train / 208 val（共 1041 crops） |
| rec dataset zip | ✅ 完成 | dataset/rec_dataset.zip（5.7 MB） |
| Colab rec notebook | ✅ 完成 | notebooks/train_rec_colab.ipynb（對齊 det：Drive checkpoint + resume A/B） |
| rec 模型訓練 | ⏳ 等待 | 需上傳 rec_dataset.zip 到 Google Drive 後跑 Colab |
| ONNX 部署測試 | 🔜 未開始 | 等訓練完成 |

---

## 原始資料來源

標注圖片位於本機：
- `C:\Users\andy_ac_chen\success\<日期>\`（已標注：success 類）
- `C:\Users\andy_ac_chen\fail\<日期>\`（已標注：fail 類）

每個日期資料夾內有：
- `*.jpg` 圖片（600×373 px）
- `Label.txt`（PPOCRLabel 格式：`日期/檔名.jpg\t[{...}]`）
- `fileState.txt`

**尚未標注的資料夾**（圖片存在但 Label.txt 為空或不存在）：
- success/20260416、20260417、20260418（部分）
- fail/20260416（部分）

可用 `tools/auto_label.py` 產生草稿，再用 PPOCRLabel 確認修正。

---

## 完整 Pipeline

```
手機端 C++ (ONNX Runtime)
    ↓ 收集大量圖片
Python 環境標注 (PPOCRLabel) 或 auto_label.py 產生草稿
    ↓ 產生 Label.txt（在各日期資料夾內）
import_labeled_data.py（匯入並 flatten 到 dataset/images/）
    ↓ dataset/images/Label.txt（合併所有標注）
split_dataset.py（分割 train/val）
    ↓ dataset/det/train/ + val/
打包 det_dataset.zip 上傳 Google Drive
    ↓
Colab GPU 訓練 (train_det_colab.ipynb)
    ↓ best_accuracy.pdparams
匯出推論模型 → 轉 ONNX
    ↓ ch_PP-OCRv4_det_infer.onnx
部署至 C++ / 手機端
```

---

## 目錄結構

```
claudeProject/
├── .claude/
│   └── settings.json              ← Claude 專案層級權限設定（Bash(*) 自動允許）
├── dataset/
│   ├── images/                    ← 已 flatten 的訓練圖片 + 合併 Label.txt
│   ├── det/
│   │   ├── train/                 ← det 訓練集圖片（744 張）
│   │   ├── val/                   ← det 驗證集圖片（186 張）
│   │   ├── train_label.txt
│   │   └── val_label.txt
│   ├── det_dataset.zip            ← 上傳 Google Drive 用（39.5 MB）
│   ├── rec/
│   │   ├── train/                 ← rec 訓練 crop（833 張）
│   │   ├── val/                   ← rec 驗證 crop（208 張）
│   │   ├── train_label.txt
│   │   └── val_label.txt
│   └── rec_dataset.zip            ← 上傳 Google Drive 用（5.7 MB）
├── configs/
│   ├── det/PP-OCRv4_mobile_det_finetune.yml
│   └── rec/PP-OCRv4_mobile_rec_finetune.yml
├── pretrained_models/             ← 預訓練權重（Colab 自動下載）
├── output/
│   ├── det/                       ← det 訓練產出
│   └── rec/                       ← rec 訓練產出
├── notebooks/
│   ├── train_det_colab.ipynb      ← Colab det 訓練（含 baseline 比較）
│   └── train_rec_colab.ipynb      ← Colab rec 訓練（未來用）
├── eval_cpp_runner/               ← C++ ONNX 推論測試
│   ├── models/                    ← ONNX 模型檔
│   └── ocr_cpp/                   ← C++ 推論原始碼
├── tools/
│   ├── import_labeled_data.py     ← ★ 匯入並合併標注資料
│   ├── auto_label.py              ← ★ 自動產生 Label.txt 草稿
│   ├── split_dataset.py           ← 分割 train/val
│   ├── prepare_rec_dataset.py     ← ★ 從 det 手動標注裁 crop 產生 rec 資料集
│   ├── run_ocr.py                 ← 本機推論（畫紅框 + JSON）
│   ├── export_model.py            ← 匯出推論模型
│   ├── convert_to_onnx.py         ← 轉換為 ONNX
│   └── download_pretrained.py     ← 下載預訓練模型
├── CLAUDE.md                      ← ★ 給 Claude 看的專案說明（本文件）
├── MANUAL.md                      ← ★ 給人看的操作手冊
└── WORKFLOW.md                    ← 完整流程說明
```

---

## 工具路徑

| 工具 | 路徑 |
|------|------|
| 本專案 | `C:\Users\andy_ac_chen\Desktop\claudeProject` |
| PaddleOCR 原始碼 | `C:\Users\andy_ac_chen\Desktop\tool\PaddleOCR` |
| PPOCRLabel 標記工具 | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel` |
| PPOCRLabel venv Python | `C:\Users\andy_ac_chen\Desktop\tool\PPOCRLabel\venv\Scripts\python.exe` |
| C++ 推論測試 | `eval_cpp_runner/build/Release/eval_runner.exe` |
| GitHub Repo | https://github.com/install88/trainingOcr |

---

## 標注規則（重要）

詳見 MANUAL.md 第一章。核心原則：
- 只標日期相關資訊（EXP / MFG / 有效期間）
- 不標地址、電話、成分、批號、時間碼
- 前綴有明確業務意義時保留（EXP 2026.01.01 → 整段標）
- 提示字在上、日期在下時，只標日期值那行
- 日期後有附加碼時，只標日期本體（20260328 AM20 → 20260328）

---

## Claude 使用注意事項

- 大多數 Bash 指令已設定為自動允許（見 `.claude/settings.json`）
- 以下操作仍需確認或已被拒絕：
  - `git push --force` / `-f`（防止強制推送）
  - `rm -rf /` 或 `rm -rf C:` 等大範圍刪除
  - `format`、`shutdown`、`reg delete` 等系統危險指令
- git push 需要在本機 CMD 手動執行（Claude 環境無 TTY，無法輸入帳密）
