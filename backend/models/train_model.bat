@echo off
echo ========================================
echo    EMOTION CLASSIFIER TRAINING
echo ========================================
echo.
echo Choose training mode:
echo 1. QUICK TRAINING (1-2 minutes) - Small dataset, faster model
echo 2. FULL TRAINING (15-30 minutes) - Full dataset, better accuracy
echo.
set /p choice="Enter your choice (1 or 2): "

cd /d "F:\internship protal\emotion_classifier\models"

if "%choice%"=="1" (
    echo.
    echo Starting QUICK TRAINING...
    echo This will take about 1-2 minutes.
    echo.
    "F:/internship protal/.venv/Scripts/python.exe" quick_train.py
) else if "%choice%"=="2" (
    echo.
    echo Starting FULL TRAINING...
    echo This will take about 15-30 minutes.
    echo.
    "F:/internship protal/.venv/Scripts/python.exe" train.py
) else (
    echo Invalid choice. Defaulting to QUICK TRAINING.
    echo.
    "F:/internship protal/.venv/Scripts/python.exe" quick_train.py
)

echo.
echo Training completed! You can now run the web app with: run_app.bat
pause