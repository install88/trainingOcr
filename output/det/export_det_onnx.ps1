$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DetOutputDir = $ScriptDir
$ProjectDir = Resolve-Path (Join-Path $ScriptDir "..\..") | Select-Object -ExpandProperty Path
$PaddleOCRDir = if ($env:PADDLEOCR_DIR) {
  $env:PADDLEOCR_DIR
} else {
  Resolve-Path (Join-Path $ProjectDir "..\tool\PaddleOCR") | Select-Object -ExpandProperty Path
}
$ConfigPath = Join-Path $ProjectDir "configs\det\PP-OCRv4_mobile_det_finetune.yml"
$BestModelPrefix = Join-Path $DetOutputDir "best_accuracy"
$InferenceDir = Join-Path $DetOutputDir "inference_legacy"
$OnnxDir = Join-Path $DetOutputDir "onnx"
$OnnxPath = Join-Path $OnnxDir "ch_PP-OCRv4_det_infer.onnx"

Write-Host "== PPOCRv4 Det export to ONNX =="
Write-Host "Project: $ProjectDir"
Write-Host ""

$requiredFiles = @(
  "$BestModelPrefix.pdparams",
  $ConfigPath,
  (Join-Path $PaddleOCRDir "tools\export_model.py")
)

foreach ($path in $requiredFiles) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing required file: $path"
  }
}

New-Item -ItemType Directory -Force -Path $InferenceDir | Out-Null
New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

Push-Location $PaddleOCRDir
try {
  Write-Host "== Step 1: Export legacy Paddle inference model =="
  python .\tools\export_model.py `
    -c $ConfigPath `
    -o Global.use_gpu=false `
       Global.export_with_pir=false `
       "Global.pretrained_model=$BestModelPrefix" `
       "Global.save_inference_dir=$InferenceDir"

  $legacyFiles = @(
    (Join-Path $InferenceDir "inference.pdmodel"),
    (Join-Path $InferenceDir "inference.pdiparams"),
    (Join-Path $InferenceDir "inference.yml")
  )

  foreach ($path in $legacyFiles) {
    if (-not (Test-Path -LiteralPath $path)) {
      throw "Legacy inference export did not create: $path"
    }
  }

  Write-Host ""
  Write-Host "== Step 2: Convert legacy inference model to ONNX =="
  paddle2onnx `
    --model_dir $InferenceDir `
    --model_filename inference.pdmodel `
    --params_filename inference.pdiparams `
    --save_file $OnnxPath `
    --opset_version 11 `
    --enable_onnx_checker True

  if (-not (Test-Path -LiteralPath $OnnxPath)) {
    throw "ONNX export did not create: $OnnxPath"
  }

  Write-Host ""
  Write-Host "== Step 3: Verify ONNX with onnxruntime =="
  python -c "import onnxruntime as ort; p=r'$OnnxPath'; s=ort.InferenceSession(p); print('OK'); print('inputs:', [(i.name,i.shape,i.type) for i in s.get_inputs()]); print('outputs:', [(o.name,o.shape,o.type) for o in s.get_outputs()])"
}
finally {
  Pop-Location
}

Write-Host ""
Write-Host "Done."
Write-Host "Legacy inference dir: $InferenceDir"
Write-Host "ONNX file: $OnnxPath"
