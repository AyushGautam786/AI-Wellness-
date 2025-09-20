@echo off
echo ========================================
echo    QUICK WEB APP TEST
echo ========================================
echo.
echo This will test if your emotion classifier works
echo.
cd /d "F:\internship protal\emotion_classifier\models"

echo Checking requirements...
if not exist "best_model.pth" (
    echo ❌ Model not found! Run train_model.bat first
    pause
    exit /b 1
)

echo ✓ Model found
echo.
echo Starting quick test...
echo The web app should open in your browser at:
echo http://localhost:7860
echo.
echo If it doesn't open automatically, copy and paste the URL above
echo.
echo Press Ctrl+C to stop the server when done testing
echo.
"F:/internship protal/.venv/Scripts/python.exe" simple_app.py