import gradio as gr
import torch
import os
import sys

print("🔄 Loading libraries...")

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
os.chdir(current_dir)

print(f"Working directory: {os.getcwd()}")

def create_simple_interface():
    """Creates a simple interface for testing"""
    
    # Check if model exists
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        def error_fn(text):
            return "❌ Model not found! Please train the model first by running train_model.bat"
        
        return gr.Interface(
            fn=error_fn,
            inputs=gr.Textbox(label="Input Text", placeholder="Model not trained yet..."),
            outputs=gr.Textbox(label="Output"),
            title="🤖 Emotion Classifier - Model Not Found",
            description="Please train the model first!"
        )
    
    # Try to load the model
    try:
        print("📥 Loading model...")
        
        # Import here to avoid loading issues
        import torch.nn as nn
        from transformers import BertTokenizer, BertModel
        
        class SimpleBertClassifier(nn.Module):
            def __init__(self, n_classes, model_name='distilbert-base-uncased', dropout=0.3):
                super(SimpleBertClassifier, self).__init__()
                from transformers import DistilBertModel
                self.bert = DistilBertModel.from_pretrained(model_name)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]
                output = self.dropout(pooled_output)
                return self.classifier(output)
        
        # Load model
        device = torch.device('cpu')  # Force CPU for stability
        model = SimpleBertClassifier(n_classes=6)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        print("✅ Model loaded successfully!")
        
        def predict_emotion(text):
            if not text.strip():
                return "Please enter some text to analyze."
            
            try:
                # Tokenize
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                # Predict
                with torch.no_grad():
                    outputs = model(encoding['input_ids'], encoding['attention_mask'])
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    
                    predicted_emotion = emotion_labels[predicted_idx]
                    confidence = probabilities[0][predicted_idx].item()
                    
                    # Format output
                    result = f"🎯 **Predicted Emotion:** {predicted_emotion.upper()}\n"
                    result += f"📊 **Confidence:** {confidence:.1%}\n\n"
                    result += "📈 **All Probabilities:**\n"
                    
                    sorted_probs = sorted(zip(emotion_labels, probabilities[0]), 
                                        key=lambda x: x[1], reverse=True)
                    
                    for emotion, prob in sorted_probs:
                        result += f"• {emotion.capitalize()}: {prob:.1%}\n"
                    
                    return result
                    
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        return gr.Interface(
            fn=predict_emotion,
            inputs=gr.Textbox(
                lines=3,
                placeholder="Enter text to analyze emotion...",
                label="📝 Input Text"
            ),
            outputs=gr.Markdown(label="🎯 Emotion Analysis"),
            title="🤖 Emotion Classifier",
            description="AI-powered emotion detection using BERT",
            examples=[
                ["I am so happy today!"],
                ["I'm really worried about tomorrow."],
                ["This makes me so angry!"],
                ["I love this new song!"]
            ]
        )
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        def error_fn(text):
            return f"❌ Error loading model: {str(e)}\n\nPlease check if you trained the model correctly."
        
        return gr.Interface(
            fn=error_fn,
            inputs=gr.Textbox(label="Input Text"),
            outputs=gr.Textbox(label="Error"),
            title="🤖 Emotion Classifier - Error",
            description=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    print("🚀 Starting Emotion Classifier Web App...")
    
    try:
        app = create_simple_interface()
        
        print("🌐 Launching web interface...")
        print("📱 App will be available at: http://localhost:7860")
        print("⏹️  Press Ctrl+C to stop")
        
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        input("Press Enter to exit...")