# AI-Wellness Application Deployment Guide

## 🚀 **QUICK LOCALHOST DEPLOYMENT**

### **Method 1: Using Batch Files (Easiest)**

1. **Double-click `start_backend.bat`** - This will:
   - Activate the Python virtual environment
   - Install required dependencies
   - Start the emotion detection API server on http://localhost:8000

2. **Double-click `start_frontend.bat`** - This will:
   - Install React dependencies 
   - Start the frontend development server on http://localhost:8080

3. **Open your browser** and visit: **http://localhost:8080**

## ☁️ **CLOUD DEPLOYMENT OPTIONS**

### **Option 1: Vercel + Railway (Recommended - Free)**

#### **Frontend (Vercel):**
1. Push your code to GitHub
2. Go to https://vercel.com and sign up
3. Import your repository
4. Select the `frontend` folder as root
5. Deploy with one click!

#### **Backend (Railway):**
1. Go to https://railway.app and sign up
2. Create new project from GitHub
3. Select the `backend` folder
4. Add environment variables
5. Deploy automatically!

### **Option 2: Netlify + Heroku**

#### **Frontend (Netlify):**
```bash
cd frontend
npm run build
# Drag and drop the 'dist' folder to Netlify
```

#### **Backend (Heroku):**
```bash
cd backend
git init
heroku create your-app-name
git add .
git commit -m "Deploy"
git push heroku main
```

### **Option 3: DigitalOcean/AWS VPS**

#### **Server Setup:**
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python
sudo apt install python3 python3-pip python3-venv nginx

# Upload your files
scp -r AI-Wellness-/ user@your-server:/home/user/
```

#### **Deploy Backend:**
```bash
cd /home/user/AI-Wellness-/backend
python3 -m venv emotion_env
source emotion_env/bin/activate
pip install fastapi uvicorn pydantic
nohup python models/mock_emotion_api.py &
```

#### **Deploy Frontend:**
```bash
cd /home/user/AI-Wellness-/frontend
npm install
npm run build
sudo cp -r dist/* /var/www/html/
```

## 🐳 **DOCKER DEPLOYMENT**

### **Create Docker Files:**

**backend/Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY models/ ./models/
COPY emotion_env/requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "models/mock_emotion_api.py"]
```

**frontend/Dockerfile:**
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

### **Deploy with Docker:**
```bash
docker-compose up --build -d
```

## 🔧 **PRODUCTION CONFIGURATION**

### **Environment Variables:**

**Frontend (.env.production):**
```env
VITE_API_URL=https://your-backend-url.com
VITE_APP_ENV=production
```

**Backend (environment):**
```env
DEBUG=False
CORS_ORIGINS=https://your-frontend-url.com
PORT=8000
```

### **Update CORS for Production:**
```python
# In mock_emotion_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📱 **MOBILE-FRIENDLY DEPLOYMENT**

### **PWA Configuration:**
Add to `frontend/public/manifest.json`:
```json
{
  "name": "AI Wellness",
  "short_name": "AIWellness",
  "display": "standalone",
  "start_url": "/",
  "theme_color": "#6366f1",
  "background_color": "#ffffff",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

## 🚀 **QUICK DEPLOYMENT STEPS**

### **Fastest Option (Vercel + Railway):**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy Frontend (Vercel):**
   - Go to vercel.com
   - Import your GitHub repo
   - Select `frontend` folder
   - Click Deploy

3. **Deploy Backend (Railway):**
   - Go to railway.app
   - Import your GitHub repo  
   - Select `backend` folder
   - Add environment variables
   - Click Deploy

4. **Update Frontend API URL:**
   - Get your Railway backend URL
   - Update `VITE_API_URL` in Vercel environment variables
   - Redeploy frontend

### **Cost Breakdown:**
- **Vercel**: Free for personal projects
- **Railway**: Free tier with 500 hours/month
- **Netlify**: Free tier available
- **Heroku**: Free tier discontinued, paid plans start at $7/month

## 📊 **MONITORING & MAINTENANCE**

### **Health Checks:**
- Backend: `GET /health`
- Frontend: Check if page loads

### **Logs:**
```bash
# Railway
railway logs

# Heroku  
heroku logs --tail

# Vercel
Check dashboard for function logs
```

## 🆘 **TROUBLESHOOTING**

### **Common Issues:**

1. **CORS Errors:**
   - Update backend CORS origins
   - Check frontend API URL

2. **Build Failures:**
   - Check Node.js version (use 18.x)
   - Verify all dependencies installed

3. **API Not Responding:**
   - Check backend health endpoint
   - Verify environment variables

### **Quick Fixes:**
```bash
# Clear npm cache
npm cache clean --force

# Rebuild frontend
rm -rf node_modules dist
npm install
npm run build

# Restart backend
kill $(lsof -t -i:8000)
python models/mock_emotion_api.py
```

## 🎉 **DEPLOYMENT SUCCESS**

Once deployed, your application will:
- ✅ Detect emotions from user text
- ✅ Generate therapeutic stories
- ✅ Provide follow-up questions
- ✅ Work on mobile devices
- ✅ Be accessible worldwide

**Example URLs after deployment:**
- Frontend: `https://ai-wellness-frontend.vercel.app`
- Backend: `https://ai-wellness-backend.railway.app`
- API Docs: `https://ai-wellness-backend.railway.app/docs`
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