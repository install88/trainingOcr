"""
convert_to_onnx.py — 將 Paddle inference 模型轉換為 ONNX 格式

轉換後的 .onnx 檔案可直接用於 C++ ONNX Runtime 推論（部署至手機端）。

前置安裝：
    pip install paddle2onnx onnxruntime

用法（Det 模型）：
    python tools/convert_to_onnx.py --model_type det

用法（Rec 模型）：
    python tools/convert_to_onnx.py --model_type rec

用法（自訂路徑）：
    python tools/convert_to_onnx.py \
        --model_dir output/det/inference \
        --save_file output/det/onnx/model.onnx

轉換完成後，將 .onnx 檔案複製到 eval_cpp_runner/models/ 即可測試。
"""

import argparse
import os
import subprocess
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "det": {
        "model_dir": os.path.join(PROJECT_DIR, "output", "det", "inference"),
        "save_file": os.path.join(PROJECT_DIR, "output", "det", "onnx", "ch_PP-OCRv4_det_infer.onnx"),
    },
    "rec": {
        "model_dir": os.path.join(PROJECT_DIR, "output", "rec", "inference"),
        "save_file": os.path.join(PROJECT_DIR, "output", "rec", "onnx", "ch_PP-OCRv4_rec_infer.onnx"),
    },
}


def check_paddle2onnx():
    """檢查 paddle2onnx 是否已安裝"""
    try:
        import paddle2onnx  # noqa: F401
        print(f"[資訊] paddle2onnx 版本：{paddle2onnx.__version__}")
        return True
    except ImportError:
        print("[錯誤] 未安裝 paddle2onnx，請執行：pip install paddle2onnx")
        return False


def check_onnxruntime():
    """檢查 onnxruntime 是否已安裝"""
    try:
        import onnxruntime as ort  # noqa: F401
        print(f"[資訊] onnxruntime 版本：{ort.__version__}")
        return True
    except ImportError:
        print("[警告] 未安裝 onnxruntime（轉換仍可執行，但無法驗證 ONNX 模型）")
        return False


def verify_onnx_model(onnx_path):
    """用 onnxruntime 驗證轉換後的 ONNX 模型"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"\n[驗證] ONNX 模型載入成功！")
        print(f"  輸入：")
        for inp in inputs:
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")
        print(f"  輸出：")
        for out in outputs:
            print(f"    - {out.name}: {out.shape} ({out.type})")
        return True
    except Exception as e:
        print(f"[警告] ONNX 模型驗證失敗：{e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Paddle inference model 轉 ONNX")
    parser.add_argument("--model_type", type=str, choices=["det", "rec"],
                        help="模型類型 (det 或 rec)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Paddle inference 模型目錄")
    parser.add_argument("--save_file", type=str, default=None,
                        help="ONNX 輸出路徑")
    parser.add_argument("--opset_version", type=int, default=11,
                        help="ONNX opset 版本（預設 11）")
    parser.add_argument("--skip_verify", action="store_true",
                        help="跳過 ONNX 模型驗證")
    return parser.parse_args()


def main():
    args = parse_args()

    if not check_paddle2onnx():
        sys.exit(1)

    has_ort = check_onnxruntime()

    if args.model_type:
        defaults = DEFAULTS[args.model_type]
        model_dir = args.model_dir or defaults["model_dir"]
        save_file = args.save_file or defaults["save_file"]
    else:
        if not args.model_dir or not args.save_file:
            print("[錯誤] 若不指定 --model_type，則必須同時提供 --model_dir 和 --save_file")
            sys.exit(1)
        model_dir = args.model_dir
        save_file = args.save_file

    pdmodel = os.path.join(model_dir, "inference.pdmodel")
    pdiparams = os.path.join(model_dir, "inference.pdiparams")

    if not os.path.exists(pdmodel):
        print(f"[錯誤] 找不到模型檔：{pdmodel}")
        print(f"  請先執行 export_model.py 匯出 inference 模型")
        sys.exit(1)

    if not os.path.exists(pdiparams):
        print(f"[錯誤] 找不到參數檔：{pdiparams}")
        sys.exit(1)

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    print(f"\n[資訊] 模型類型：{args.model_type or '自訂'}")
    print(f"[資訊] 來源目錄：{model_dir}")
    print(f"[資訊] 輸出路徑：{save_file}")
    print(f"[資訊] ONNX opset：{args.opset_version}")
    print()

    cmd = [
        sys.executable, "-m", "paddle2onnx.convert",
        "--model_dir", model_dir,
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", save_file,
        "--opset_version", str(args.opset_version),
        "--enable_onnx_checker", "True",
    ]

    print(f"[執行] {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[錯誤] 轉換失敗（exit code: {result.returncode}）")
        sys.exit(result.returncode)

    print(f"\n[完成] ONNX 模型已儲存至：{save_file}")

    if has_ort and not args.skip_verify:
        verify_onnx_model(save_file)

    # 提示後續操作
    cpp_models_dir = os.path.join(PROJECT_DIR, "eval_cpp_runner", "models")
    if os.path.isdir(cpp_models_dir):
        print(f"\n[提示] 你可以將 ONNX 模型複製到 C++ 推論目錄進行測試：")
        print(f"  copy \"{save_file}\" \"{cpp_models_dir}\\\"")


if __name__ == "__main__":
    main()
