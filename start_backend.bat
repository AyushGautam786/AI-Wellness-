@echo off
echo ===================================
echo  Starting AI-Wellness Backend
echo ===================================
echo.
echo Activating virtual environment...
cd /d "f:\internship protal\AI-Wellness-\backend"
call ".\emotion_env\Scripts\activate.bat"

echo.
echo Installing dependencies if needed...
pip install fastapi uvicorn torch transformers numpy scikit-learn

echo.
echo Starting FastAPI server...
cd models
python emotion_api.py

pause