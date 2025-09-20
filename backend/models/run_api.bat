@echo off
echo Starting Emotion Detection API Service...
echo.
echo Make sure you have installed the requirements:
echo pip install -r api_requirements.txt
echo.
echo API will be available at:
echo - Main endpoint: http://127.0.0.1:8000
echo - Documentation: http://127.0.0.1:8000/docs
echo - Interactive testing: http://127.0.0.1:8000/redoc
echo.
echo Press Ctrl+C to stop the server
echo.
python emotion_api.py
pause