import gradio as gr
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os

# --- 1. Model Architecture ---
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
        from transformers import DistilBertModel
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
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first by running: python train.py or python quick_train.py")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the trained weights
        try:
            print(f"Loading model from: {model_path}")
            
            # Load model architecture - try both model types
            try:
                # First try to load as SimpleBertClassifier (from quick_train)
                self.model = SimpleBertClassifier(n_classes=6)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded SimpleBertClassifier model successfully!")
            except:
                try:
                    # Fallback to BertEmotionClassifier (from regular train)
                    self.model = BertEmotionClassifier(n_classes=6)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded BertEmotionClassifier model successfully!")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise
                    
        except Exception as e:
            print(f"Error loading model: {e}")
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
                    emotion: prob.item()
                    for emotion, prob in zip(self.emotion_labels, probabilities[0])
                }

            return {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'predicted_emotion': 'error',
                'confidence': 0.0,
                'all_probabilities': {}
            }

# --- 3. Gradio Web Interface ---
def create_web_interface():
    """Creates and configures the Gradio web interface."""
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
        error_msg = "Model file not found. Please train the model first by running 'python train.py'"
        print(error_msg)
        
        def error_interface(text):
            return f"**Error:** {error_msg}"
        
        interface = gr.Interface(
            fn=error_interface,
            inputs=gr.Textbox(
                lines=5,
                placeholder="Model not found. Please train first.",
                label="Input Text"
            ),
            outputs=gr.Markdown(label="Error Message"),
            title="📝 Text Emotion Classification - Model Not Found",
            description="Please train the model first by running 'python train.py'",
        )
        return interface
    
    # Load the trained model predictor
    try:
        predictor = EmotionPredictor(model_path)
    except Exception as e:
        def error_interface(text):
            return f"**Error loading model:** {str(e)}"
        
        interface = gr.Interface(
            fn=error_interface,
            inputs=gr.Textbox(
                lines=5,
                placeholder="Error loading model.",
                label="Input Text"
            ),
            outputs=gr.Markdown(label="Error Message"),
            title="📝 Text Emotion Classification - Model Error",
            description=f"Error loading model: {str(e)}",
        )
        return interface

    def predict_emotion_for_gradio(text):
        """Wrapper function for Gradio interface."""
        if not text or not text.strip():
            return "Please enter some text to analyze."

        result = predictor.predict(text)
        
        if result['predicted_emotion'] == 'error':
            return "Error occurred during prediction. Please try again."
        
        # Format the output as Markdown for better display
        output = f"**Predicted Emotion:** {result['predicted_emotion'].upper()}\n"
        output += f"**Confidence:** {result['confidence']:.2%}\n\n"
        output += "**All Probabilities:**\n"
        
        # Sort probabilities for clearer presentation
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda item: item[1], reverse=True)
        
        for emotion, prob in sorted_probs:
            output += f"- **{emotion.capitalize()}:** {prob:.2%}\n"
            
        return output

    # Create and return the Gradio interface object
    interface = gr.Interface(
        fn=predict_emotion_for_gradio,
        inputs=gr.Textbox(
            lines=5,
            placeholder="Enter text here to analyze its emotion...",
            label="Input Text"
        ),
        outputs=gr.Markdown(label="Emotion Analysis Results"),
        title="📝 Text Emotion Classification",
        description="Enter any text and the model will predict its emotion. The model is based on BERT and was fine-tuned on the 'Emotion' dataset.",
        examples=[
            ["I am so incredibly happy about my promotion at work!"],
            ["I'm really worried about the final exam tomorrow."],
            ["I can't believe you would say that to me, I'm furious!"],
        ]
    )
    return interface

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("Initializing Gradio web interface...")
    try:
        web_app = create_web_interface()
        print("Launching Gradio web interface on localhost...")
        print("🌐 Starting server...")
        print("📱 The app will be available at: http://127.0.0.1:7860")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        web_app.launch(
            server_name="0.0.0.0",       # Listen on all interfaces
            server_port=7860,            # Default Gradio port
            share=False,                 # Don't create public link
            debug=True,                  # Enable debug mode
            show_error=True,             # Show error messages
            inbrowser=True               # Auto-open browser
        )
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install gradio torch transformers")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you trained the model first")
        print("2. Check if port 7860 is available")
        print("3. Try running: python app.py directly")
        input("Press Enter to exit...")