"""
FastAPI-based emotion detection service for integration with story generation app.
This service provides REST API endpoints for emotion detection using the trained BERT model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertModel
import os
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Model Architecture (Same as app.py) ---
class BertEmotionClassifier(nn.Module):
    """BERT-based classifier model."""
    def __init__(self, n_classes, model_name='bert-base-uncased', dropout=0.3):
        super(BertEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class SimpleBertClassifier(nn.Module):
    """Simplified BERT classifier for quick training."""
    def __init__(self, n_classes, model_name='distilbert-base-uncased', dropout=0.3):
        super(SimpleBertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
        output = self.dropout(pooled_output)
        return self.classifier(output)

# --- 2. Emotion Predictor Class ---
class EmotionPredictor:
    """Handles loading the model and making predictions."""
    def __init__(self, model_path, tokenizer_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the trained weights
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Load model architecture - try both model types
            try:
                # First try to load as SimpleBertClassifier (from quick_train)
                self.model = SimpleBertClassifier(n_classes=6)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded SimpleBertClassifier model successfully!")
            except:
                try:
                    # Fallback to BertEmotionClassifier (from regular train)
                    self.model = BertEmotionClassifier(n_classes=6)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Loaded BertEmotionClassifier model successfully!")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    def predict(self, text):
        """Predicts emotion from a single text string."""
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                
                predicted_emotion = self.emotion_labels[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item()

                # Get all probabilities
                all_probabilities = {
                    emotion: round(prob.item(), 4)
                    for emotion, prob in zip(self.emotion_labels, probabilities[0])
                }

            return {
                'predicted_emotion': predicted_emotion,
                'confidence': round(confidence, 4),
                'all_probabilities': all_probabilities,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'predicted_emotion': 'error',
                'confidence': 0.0,
                'all_probabilities': {},
                'success': False,
                'error': str(e)
            }

# --- 3. Pydantic Models for API ---
class TextInput(BaseModel):
    text: str
    
class EmotionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

# --- 4. FastAPI Application ---
app = FastAPI(
    title="Emotion Detection API",
    description="REST API for emotion detection using BERT model",
    version="1.0.0"
)

# Configure CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the emotion predictor on startup."""
    global predictor
    
    # Try multiple possible model paths
    possible_model_paths = [
        './models/best_model.pth',
        './best_model.pth',
        'best_model.pth',
        os.path.join(os.path.dirname(__file__), 'best_model.pth'),
        os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        logger.error("Model file not found. Please train the model first.")
        return
    
    try:
        predictor = EmotionPredictor(model_path)
        logger.info("Emotion predictor loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load emotion predictor: {e}")

# --- 5. API Endpoints ---
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="running",
        message="Emotion Detection API is running",
        model_loaded=predictor is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        message="Model loaded successfully" if predictor is not None else "Model not loaded",
        model_loaded=predictor is not None
    )

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(text_input: TextInput):
    """Predict emotion from input text."""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if not text_input.text or not text_input.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text input cannot be empty"
        )
    
    try:
        result = predictor.predict(text_input.text.strip())
        return EmotionResponse(**result)
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
        "emotions": ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
        "total_count": 6
    }

# --- 6. Run Application ---
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Emotion Detection API...")
    print("📝 API Documentation will be available at: http://127.0.0.1:8000/docs")
    print("🔧 Interactive API testing at: http://127.0.0.1:8000/redoc")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )