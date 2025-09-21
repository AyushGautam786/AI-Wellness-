#!/bin/bash
echo "🚀 Starting AI-Wellness API..."
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la

echo "Starting FastAPI server..."
exec python3 main.py