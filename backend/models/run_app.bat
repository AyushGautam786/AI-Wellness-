@echo off
echo ========================================
echo    EMOTION CLASSIFIER WEB APP
echo ========================================
echo.
echo Starting Emotion Classifier Web App...
echo.
cd /d "F:\internship protal\emotion_classifier\models"
echo Current directory: %CD%
echo.
echo Checking if model file exists...
if exist "best_model.pth" (
    echo ✓ Model file found!
) else (
    echo ❌ Model file not found! Please train the model first.
    echo Run train_model.bat to train the model.
    pause
    exit /b 1
)
echo.
echo Starting web server...
echo The app will be available at: http://localhost:7860
echo.
echo Please wait for the server to start...
echo Press Ctrl+C to stop the server
echo.
"F:/internship protal/.venv/Scripts/python.exe" "%~dp0simple_app.py"
pause