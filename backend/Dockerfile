FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY models/api_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy the application code
COPY models/ .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "mock_emotion_api.py"]