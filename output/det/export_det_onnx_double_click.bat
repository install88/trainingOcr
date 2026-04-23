@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%export_det_onnx.ps1"

echo PPOCRv4 Det export to ONNX
echo.

if not exist "%PS_SCRIPT%" (
  echo [ERROR] PowerShell script not found:
  echo %PS_SCRIPT%
  echo.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo [OK] Finished.
) else (
  echo [ERROR] Failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
