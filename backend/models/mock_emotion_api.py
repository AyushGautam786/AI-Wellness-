"""
Simple Mock Emotion API for Railway deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI-Wellness Emotion Detection API",
    description="Emotion detection API for AI-Wellness therapeutic application",
    version="1.0.0"
)

# --- CORS Configuration for Production ---
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:4173",
    "https://*.vercel.app",
    "https://ai-wellness-app.vercel.app",  # Update with your actual Vercel URL
    "https://*.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    success: bool
    message: str

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

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(input_data: TextInput):
    """Emotion prediction endpoint with enhanced keyword detection."""
    try:
        text = input_data.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Enhanced keyword-based emotion detection
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
            # Default to neutral/mixed emotions
            emotion = random.choice(['joy', 'sadness'])  # Bias toward common emotions
            confidence = random.uniform(0.45, 0.65)
        
        logger.info(f"Emotion detected: {emotion} (confidence: {confidence:.2f}) for text: '{text[:50]}...'")
        
        return EmotionResponse(
            predicted_emotion=emotion,
            confidence=round(confidence, 2),
            success=True,
            message=f"Emotion '{emotion}' detected with {confidence:.2f} confidence"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

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
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )