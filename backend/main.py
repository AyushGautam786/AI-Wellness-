#!/usr/bin/env python
"""
Railway deployment entry point for AI-Wellness backend
"""
import os

print("🚀 Starting AI-Wellness Emotion Detection API...")

try:
    # Import the FastAPI app from the root directory
    from app import app
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
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Files in current directory:")
    for file in os.listdir("."):
        print(f"  - {file}")
    raise