# AI-Wellness Cloud Deployment Quick Commands

## ☁️ Vercel Deployment (Frontend)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy frontend
cd frontend
vercel

# Set environment variables in Vercel dashboard:
# VITE_API_URL = https://your-backend-url.railway.app
```

## 🚂 Railway Deployment (Backend)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy backend
cd backend
railway deploy

# Your backend will be available at:
# https://your-app-name.railway.app
```

## 🔗 Complete Deployment Flow

1. **Deploy Backend First:**
   ```bash
   cd backend
   railway deploy
   # Note down the Railway URL
   ```

2. **Deploy Frontend with Backend URL:**
   ```bash
   cd frontend
   # Update .env.production with Railway URL
   echo "VITE_API_URL=https://your-backend.railway.app" > .env.production
   vercel --prod
   ```

3. **Test Deployment:**
   ```bash
   curl https://your-backend.railway.app/health
   curl https://your-frontend.vercel.app
   ```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Your app will be available at:
# Frontend: http://localhost:80
# Backend: http://localhost:8000
```

## 📱 Mobile PWA Setup

```bash
# Add to frontend/public/manifest.json
# Add service worker for offline support
# Test PWA features on mobile devices
```

## 🚀 One-Click Deploy Links

### Frontend Options:
- **Vercel**: [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)
- **Netlify**: [![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start)

### Backend Options:
- **Railway**: [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)
- **Heroku**: [![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## 💡 Pro Tips

1. **Environment Variables**: Always set production API URLs
2. **CORS**: Update backend CORS origins for production domains
3. **Monitoring**: Set up health checks and alerts
4. **SSL**: All cloud platforms provide HTTPS automatically
5. **Caching**: Enable caching for better performance

## 📊 Deployment Costs

| Platform | Frontend | Backend | Total/Month |
|----------|----------|---------|-------------|
| Vercel + Railway | Free | Free | $0 |
| Netlify + Heroku | Free | $7+ | $7+ |
| DigitalOcean | $5 | $5 | $10 |
| AWS | $1-5 | $5-10 | $6-15 |

**Recommended**: Start with Vercel + Railway (free) for testing, upgrade as needed.

## nixpacks.toml for Backend

```toml
# Create nixpacks.toml file instead of Dockerfile
[phases.setup]
nixPkgs = ["python311"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "python models/mock_emotion_api.py"

[variables]
PORT = "8000"
```

#!/usr/bin/env python3
"""
Entry point for Railway deployment
"""
import os
import sys

# Add models directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Import and run the FastAPI app
from mock_emotion_api import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)