# 🤖 Emotion Classifier

A BERT-based emotion classification system with a user-friendly web interface. This project can predict emotions from text input with high accuracy using state-of-the-art transformer models.

![Emotion Classification Demo](https://img.shields.io/badge/Status-Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Gradio](https://img.shields.io/badge/Interface-Gradio-orange)

## 🎯 Features

- **6 Emotion Categories**: Sadness, Joy, Love, Anger, Fear, Surprise
- **BERT-based Model**: Uses transformer architecture for high accuracy
- **Web Interface**: Easy-to-use Gradio-based web application
- **Quick Training**: Fast training option for testing (1-2 minutes)
- **Full Training**: Complete training for production use (15-30 minutes)
- **CPU Optimized**: Works efficiently on CPU-only systems

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AyushGautam786/emotion_classifier.git
cd emotion_classifier
```

### 2. Install Dependencies
```bash
pip install -r models/requirements.txt
```

### 3. Train the Model (Required)
```bash
# Quick training (1-2 minutes) - REQUIRED FIRST STEP
cd models
double-click train_model.bat
# Choose option 1 for quick training

# Note: The trained model (best_model.pth) is not included in the repository
# due to GitHub's file size limits. You must train it locally first!
```

### 4. Run the Web App
```bash
# Start the web interface
double-click run_app.bat
# Or run: python simple_app.py
```

### 5. Access the App
Open your browser and go to: `http://localhost:7860`

## 📁 Project Structure

```
emotion_classifier/
├── models/
│   ├── app.py              # Main Gradio web application
│   ├── simple_app.py       # Simplified web app (recommended)
│   ├── train.py            # Full training script
│   ├── quick_train.py      # Quick training script
│   ├── launcher.py         # Cross-platform launcher
│   ├── requirements.txt    # Python dependencies
│   ├── train_model.bat     # Training launcher (Windows)
│   ├── run_app.bat         # App launcher (Windows)
│   ├── quick_test.bat      # Quick test launcher
│   └── test_setup.py       # Diagnostic test script
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🎮 Usage Examples

### Web Interface
1. Start the web app: `python simple_app.py`
2. Enter text in the input box
3. View emotion predictions with confidence scores

### Example Predictions
- **"I am so happy today!"** → **JOY** (85% confidence)
- **"I'm really worried about tomorrow"** → **FEAR** (78% confidence)
- **"This makes me so angry!"** → **ANGER** (92% confidence)
- **"I love this new song!"** → **LOVE** (88% confidence)

## 🛠️ Training Options

### Quick Training (Recommended for Testing)
- **Time**: 1-2 minutes
- **Dataset**: 500 samples
- **Model**: DistilBERT (lightweight)
- **Accuracy**: ~70-80%
- **Use**: Testing and development

### Full Training (Production)
- **Time**: 15-30 minutes
- **Dataset**: 2000+ samples
- **Model**: BERT-base
- **Accuracy**: ~85-90%
- **Use**: Production deployment

## 📊 Model Performance

| Training Mode | Time | Samples | Model | Accuracy |
|---------------|------|---------|-------|----------|
| Quick | 1-2 min | 500 | DistilBERT | 70-80% |
| Optimized | 5-10 min | 2k | BERT | 80-85% |
| Full | 30+ min | 16k | BERT | 85-90% |

## 🔧 Technical Details

### Model Architecture
- **Base Model**: BERT/DistilBERT
- **Classification Head**: Linear layer with dropout
- **Output**: 6 emotion classes
- **Input**: Text sequences (max 128 tokens)

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 to 5e-5
- **Batch Size**: 16-64
- **Loss Function**: CrossEntropyLoss

### Dataset
- **Source**: Hugging Face "emotion" dataset
- **Size**: 16,000 training samples
- **Languages**: English
- **Labels**: 6 emotion categories

## 🌐 Web Interface Features

- **Real-time Prediction**: Instant emotion analysis
- **Confidence Scores**: Shows probability for each emotion
- **Example Texts**: Pre-loaded examples to try
- **Error Handling**: Graceful error messages
- **Responsive Design**: Works on desktop and mobile

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**
   ```bash
   # Solution: Train the model first
   python quick_train.py
   ```

2. **Import errors**
   ```bash
   # Solution: Install dependencies
   pip install -r requirements.txt
   ```

3. **Port already in use**
   ```
   # Solution: The app will show available ports
   # Or manually change port in the code
   ```

4. **Memory issues**
   ```
   # Solution: Use quick training mode
   # Reduce batch size in config
   ```

### Diagnostic Test
```bash
python test_setup.py
```

## 📦 Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: BERT model implementation
- **Gradio**: Web interface framework
- **Datasets**: Hugging Face datasets
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scikit-learn**: ML utilities
- **tqdm**: Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and datasets
- **Google** for the BERT model
- **Gradio** for the web interface framework
- **PyTorch** team for the deep learning framework

## 📞 Contact

- **Author**: Ayush Gautam
- **GitHub**: [@AyushGautam786](https://github.com/AyushGautam786)
- **Repository**: [emotion_classifier](https://github.com/AyushGautam786/emotion_classifier)

## 🎯 Future Improvements

- [ ] Multi-language support
- [ ] Real-time audio emotion detection
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Model fine-tuning interface
- [ ] Batch processing capability
- [ ] Export predictions to CSV

---

**⭐ If you find this project helpful, please give it a star!**