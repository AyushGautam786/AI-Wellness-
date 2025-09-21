"""
Ultra-simple FastAPI for Railway deployment without pydantic models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import random
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI-Wellness Emotion Detection API",
    description="Simple emotion detection API",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock emotion prediction ---
EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@app.get("/")
async def root():
    return {
        "message": "AI-Wellness Emotion Detection API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ai-wellness-emotion-api",
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "production")
    }

@app.post("/predict")
async def predict_emotion(request: dict):
    """Simple emotion prediction endpoint."""
    try:
        # Get text from request
        text = request.get('text', '').strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Simple keyword-based emotion detection
        text_lower = text.lower()
        
        # Joy keywords
        if any(word in text_lower for word in [
            'happy', 'joy', 'great', 'wonderful', 'amazing', 'excited', 
            'fantastic', 'awesome', 'delighted', 'thrilled', 'cheerful'
        ]):
            emotion = 'joy'
            confidence = random.uniform(0.75, 0.95)
            
        # Sadness keywords  
        elif any(word in text_lower for word in [
            'sad', 'depressed', 'down', 'upset', 'disappointed', 'hurt',
            'gloomy', 'miserable', 'heartbroken', 'melancholy'
        ]):
            emotion = 'sadness'
            confidence = random.uniform(0.70, 0.90)
            
        # Love keywords
        elif any(word in text_lower for word in [
            'love', 'adore', 'cherish', 'care', 'affection', 'devoted',
            'passionate', 'romantic', 'tender', 'fond'
        ]):
            emotion = 'love'
            confidence = random.uniform(0.70, 0.85)
            
        # Anger keywords
        elif any(word in text_lower for word in [
            'angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage',
            'frustrated', 'outraged', 'livid', 'hostile'
        ]):
            emotion = 'anger'
            confidence = random.uniform(0.75, 0.90)
            
        # Fear keywords
        elif any(word in text_lower for word in [
            'scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified',
            'panic', 'frightened', 'concerned', 'apprehensive'
        ]):
            emotion = 'fear'
            confidence = random.uniform(0.70, 0.85)
            
        # Surprise keywords
        elif any(word in text_lower for word in [
            'surprised', 'shocked', 'amazed', 'unexpected', 'astonished',
            'stunned', 'bewildered', 'startled'
        ]):
            emotion = 'surprise'
            confidence = random.uniform(0.65, 0.80)
        else:
            # Default emotions
            emotion = random.choice(['joy', 'sadness'])
            confidence = random.uniform(0.45, 0.65)
        
        logger.info(f"Emotion detected: {emotion} (confidence: {confidence:.2f})")
        
        return {
            "predicted_emotion": emotion,
            "confidence": round(confidence, 2),
            "success": True,
            "message": f"Emotion '{emotion}' detected with {confidence:.2f} confidence"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/emotions")
async def get_supported_emotions():
    """Get list of supported emotions."""
    return {
        "emotions": EMOTIONS,
        "total_count": len(EMOTIONS),
        "service": "ai-wellness-emotion-api"
    }

# --- Run Application ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"Starting AI-Wellness Emotion API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")