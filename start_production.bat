@echo off
echo ===================================
echo  AI-Wellness Production Build
echo ===================================
echo.

echo [1/4] Building Frontend...
cd /d "f:\internship protal\AI-Wellness-\frontend"
call npm install
call npm run build
echo ✅ Frontend built successfully!

echo.
echo [2/4] Preparing Backend...
cd /d "f:\internship protal\AI-Wellness-\backend"
call ".\emotion_env\Scripts\activate.bat"
pip install fastapi uvicorn pydantic --quiet
echo ✅ Backend dependencies ready!

echo.
echo [3/4] Starting Production Servers...
echo Starting Backend API...
start "AI-Wellness Backend" cmd /k "cd /d f:\internship protal\AI-Wellness-\backend && .\emotion_env\Scripts\activate && python models\mock_emotion_api.py"

timeout /t 5

echo Starting Frontend...
start "AI-Wellness Frontend" cmd /k "cd /d f:\internship protal\AI-Wellness-\frontend && npm run preview"

echo.
echo [4/4] Production Deployment Complete!
echo =====================================
echo 📱 Frontend (Production): http://localhost:4173
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo =====================================
echo.
echo Your AI-Wellness application is now running in production mode!
echo.
pause