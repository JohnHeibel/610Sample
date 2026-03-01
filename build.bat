@echo off
echo === Building attention_cuda extension ===
python setup.py build_ext --inplace
if %errorlevel% neq 0 (
    echo BUILD FAILED
    exit /b 1
)
echo.
echo === Build complete ===
dir attention_cuda*.pyd 2>nul
