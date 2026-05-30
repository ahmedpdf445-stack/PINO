@echo off
setlocal
if not exist pino_kernel.cu (
    echo ERROR: pino_kernel.cu not found in the current directory.
    exit /b 1
)

nvcc -O3 -std=c++17 pino_kernel.cu -lcufft -o pino_kernel.dll
if errorlevel 1 (
    echo ERROR: CUDA build failed.
    exit /b 1
)

echo Build complete: pino_kernel.dll
