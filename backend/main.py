#!/usr/bin/env python3
"""
Railway deployment entry point for AI-Wellness backend
"""
import os
import sys

# Add models directory to Python path
models_path = os.path.join(os.path.dirname(__file__), 'models')
sys.path.insert(0, models_path)

print("Starting AI-Wellness Emotion Detection API...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Models path: {models_path}")

try:
    # Import the FastAPI app
    from mock_emotion_api import app
    print("✅ Successfully imported FastAPI app")
    
    # Start the server
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available files in models directory:")
    try:
        for file in os.listdir(models_path):
            print(f"  - {file}")
    except FileNotFoundError:
        print("  Models directory not found!")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)