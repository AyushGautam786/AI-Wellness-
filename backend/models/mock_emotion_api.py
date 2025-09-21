"""
Simple Mock Emotion API for testing the frontend integration
This bypasses the PyTorch model loading issue and provides mock responses
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
    title="Mock Emotion Detection API",
    description="Mock API for emotion detection while debugging the ML model",
    version="1.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:4173",
    "https://*.vercel.app",  # Allow all Vercel apps
    "https://your-vercel-app.vercel.app",  # We'll update this after Vercel deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    return {"message": "Mock Emotion Detection API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mock-emotion-api"}

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(input_data: TextInput):
    """Mock emotion prediction endpoint."""
    try:
        text = input_data.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Mock logic: Simple keyword-based emotion detection
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['happy', 'joy', 'great', 'wonderful', 'amazing', 'excited']):
            emotion = 'joy'
            confidence = random.uniform(0.75, 0.95)
        elif any(word in text_lower for word in ['sad', 'depressed', 'down', 'upset', 'disappointed']):
            emotion = 'sadness'
            confidence = random.uniform(0.70, 0.90)
        elif any(word in text_lower for word in ['love', 'adore', 'cherish', 'care', 'affection']):
            emotion = 'love'
            confidence = random.uniform(0.70, 0.85)
        elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'irritated', 'annoyed']):
            emotion = 'anger'
            confidence = random.uniform(0.75, 0.90)
        elif any(word in text_lower for word in ['scared', 'afraid', 'worried', 'anxious', 'nervous']):
            emotion = 'fear'
            confidence = random.uniform(0.70, 0.85)
        elif any(word in text_lower for word in ['surprised', 'shocked', 'amazed', 'unexpected']):
            emotion = 'surprise'
            confidence = random.uniform(0.65, 0.80)
        else:
            # Random emotion for texts that don't match keywords
            emotion = random.choice(EMOTIONS)
            confidence = random.uniform(0.50, 0.75)
        
        return EmotionResponse(
            predicted_emotion=emotion,
            confidence=confidence,
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
        "total_count": len(EMOTIONS)
    }

# --- Run Application ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)