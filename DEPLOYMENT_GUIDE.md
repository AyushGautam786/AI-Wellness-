# AI-Wellness Emotion Detection & Story Generation System

## 🚀 **QUICK LOCALHOST DEPLOYMENT**

### **Method 1: Using Batch Files (Easiest)**

1. **Double-click `start_backend.bat`** - This will:
   - Activate the Python virtual environment
   - Install required dependencies
   - Start the emotion detection API server on http://localhost:8000

2. **Double-click `start_frontend.bat`** - This will:
   - Install React dependencies 
   - Start the frontend development server on http://localhost:5173

3. **Open your browser** and visit: **http://localhost:5173**

### **Method 2: Manual Commands**

**Terminal 1 (Backend):**
```powershell
cd "f:\internship protal\AI-Wellness-\backend"
& ".\emotion_env\Scripts\Activate.ps1"
pip install fastapi uvicorn torch transformers numpy scikit-learn
cd models
python emotion_api.py
```

**Terminal 2 (Frontend):**
```powershell
cd "f:\internship protal\AI-Wellness-\frontend"  
npm install
npm run dev
```

## ✅ **SUCCESS INDICATORS**

### **Backend Running Successfully:**
- Console shows: `🚀 Starting Emotion Detection API...`
- Visit http://localhost:8000/health shows: `"model_loaded": true`
- API docs available at: http://localhost:8000/docs

### **Frontend Running Successfully:**
- Console shows: `Local: http://localhost:5173/`
- Chat interface loads without errors
- Real-time emotion detection works as you type

## 🧪 **Testing the System**

1. **Open** http://localhost:5173
2. **Type** a message like "I'm feeling really excited today!"
3. **Watch** for real-time emotion detection 
4. **See** therapeutic story recommendations appear
5. **Check** follow-up questions for reflection

## 🛠️ **Troubleshooting**

| Issue | Solution |
|-------|----------|
| Backend won't start | Run `pip install fastapi uvicorn torch transformers` |
| Port 8000 in use | Change port in `backend/models/emotion_api.py` |
| Frontend errors | Delete `node_modules`, run `npm install` again |
| Model not found | Check that `best_model.pth` exists in `backend/models/` |

## 📁 **Project Structure**
```
AI-Wellness-/
├── start_backend.bat      # Easy backend startup
├── start_frontend.bat     # Easy frontend startup  
├── backend/               # Python emotion detection API
│   ├── emotion_env/       # Virtual environment
│   └── models/            # ML model and API code
├── frontend/              # React chat application
└── dataset/               # Therapeutic story database
```

## 🎯 **Features**
- **6 Emotion Detection**: Joy, Sadness, Love, Anger, Fear, Surprise
- **36 Therapeutic Stories**: 6 stories per emotion with clinical elements
- **Real-time Analysis**: Emotion detection as you type
- **Interactive Chat**: Responsive interface with story recommendations
- **REST API**: FastAPI backend for easy integration

---

**Your AI-Wellness emotion detection system is now ready for localhost deployment!** 🎉