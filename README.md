# H2S Project - Emotion-Aware Story Generation System

## Overview
The H2S (Heart-to-Story) project combines advanced machine learning emotion detection with therapeutic storytelling to provide personalized mental wellness support. Users can express their thoughts and feelings, and the system responds with emotionally-appropriate therapeutic stories designed to provide comfort, insight, and healing.

## 🎯 Features
- **Real-time Emotion Detection**: BERT/DistilBERT-powered ML model analyzes text for 6 core emotions
- **Therapeutic Story Generation**: 36 professionally-crafted stories with therapeutic elements
- **Interactive Chat Interface**: Responsive React application with emotion visualization
- **RESTful API**: FastAPI backend for seamless integration
- **Comprehensive Dataset**: Clinically-informed therapeutic narratives

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- Git

### Installation

1. **Clone and Navigate**
```bash
cd "f:\internship protal\AI-Wellness-"
```

2. **Backend Setup**
```bash
cd backend
python -m venv emotion_env
# Windows:
.\emotion_env\Scripts\activate
# macOS/Linux:
# source emotion_env/bin/activate

pip install -r requirements.txt
python test_setup.py
```

3. **Frontend Setup** (new terminal)
```bash
cd "f:\internship protal\h2s-project\frontend"
npm install
```

4. **Start Backend Server**
```bash
cd "f:\internship protal\h2s-project\backend"
# Ensure virtual environment is activated
python emotion_api.py
# Server starts at http://localhost:8000
```

5. **Start Frontend Application** (new terminal)
```bash
cd "f:\internship protal\h2s-project\frontend"
npm run dev
# App starts at http://localhost:5173
```

## 📁 Project Structure

```
h2s-project/
├── backend/                 # Python ML backend
│   ├── emotion_api.py      # FastAPI server
│   ├── train.py            # Model training
│   ├── best_model.pth      # Pre-trained model
│   ├── test_setup.py       # Environment validation
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React application
│   ├── src/
│   │   ├── components/
│   │   │   └── EmotionAwareChatInterface.tsx
│   │   ├── lib/
│   │   │   ├── emotionStoryMapping.ts
│   │   │   └── emotionApiService.ts
│   │   └── ...
│   └── package.json
│
├── dataset/                # Therapeutic stories
│   ├── emotions_stories_part1.json  # Joy, Sadness
│   ├── emotions_stories_part2.json  # Love, Anger  
│   ├── emotions_stories_part3.json  # Fear, Surprise
│   └── ...
│
└── README.md
```

## 🧠 Supported Emotions

| Emotion | Description | Story Themes |
|---------|-------------|--------------|
| **Joy** | Happiness, celebration, contentment | Gratitude, sharing happiness, maintaining positivity |
| **Sadness** | Grief, loss, melancholy | Healing, finding meaning, processing loss |
| **Love** | Affection, connection, care | Relationships, self-love, community bonds |
| **Anger** | Frustration, injustice, intensity | Boundaries, constructive channeling, advocacy |
| **Fear** | Anxiety, uncertainty, caution | Courage building, gradual exposure, wisdom in caution |
| **Surprise** | Unexpected events, discovery | Adaptability, openness, embracing change |

## 🔧 API Usage

### Emotion Detection Endpoint
```bash
POST http://localhost:8000/detect-emotion
Content-Type: application/json

{
  "text": "I'm feeling really excited about this new project!"
}
```

**Response:**
```json
{
  "emotion": "joy",
  "confidence": 0.95,
  "all_scores": {
    "joy": 0.95,
    "sadness": 0.01,
    "love": 0.02,
    "anger": 0.01,
    "fear": 0.01,
    "surprise": 0.00
  }
}
```

### Health Check
```bash
GET http://localhost:8000/health
```

## 🧪 Testing Your Setup

1. **Test Backend**:
```bash
curl -X POST http://localhost:8000/detect-emotion -H "Content-Type: application/json" -d "{\"text\": \"I am so happy today!\"}"
```

2. **Test Frontend**: 
   - Open http://localhost:5173
   - Type a message expressing emotion
   - Observe real-time emotion detection and story generation

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Change port in `emotion_api.py` or kill existing process |
| Model file missing | Run `train.py` to generate `best_model.pth` |
| Dependency conflicts | Use virtual environment, install exact versions |
| API connection failed | Verify backend running on localhost:8000 |

## 📊 Dataset Information

The therapeutic story dataset contains:
- **36 total stories** (6 per emotion)
- **Clinical elements**: CBT techniques, narrative therapy, mindfulness
- **Follow-up questions**: Designed to encourage reflection
- **Therapeutic goals**: Specific mental wellness objectives per emotion

## 🔮 Future Enhancements

- [ ] User progress tracking and story history
- [ ] Emotion trend analysis over time
- [ ] Voice input for accessibility  
- [ ] Professional dashboard integration
- [ ] Multi-language support
- [ ] Advanced personalization algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add therapeutic stories following the JSON schema
4. Test emotion detection accuracy
5. Submit pull request with detailed description

## 📝 License

This project is developed for educational and therapeutic purposes. Please ensure appropriate clinical oversight when used in professional mental health settings.

## 📞 Support

For technical issues:
1. Check troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure both frontend and backend servers are running
4. Test API endpoints individually

---

**Made with ❤️ for mental wellness and emotional support**