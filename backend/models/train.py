import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm
import warnings
import os

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
def load_text_data():
    """Load text emotion dataset from Hugging Face and prepare it."""
    print("Loading 'emotion' dataset from Hugging Face...")
    try:
        # Load emotion dataset
        dataset = load_dataset("emotion")
        print("Dataset loaded successfully from Hugging Face!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and datasets library installed.")
        raise

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Map labels to human-readable names
    emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    train_df['emotion_name'] = train_df['label'].map(emotion_mapping)
    val_df['emotion_name'] = val_df['label'].map(emotion_mapping)
    test_df['emotion_name'] = test_df['label'].map(emotion_mapping)
    
    print(f"Dataset sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

# --- 2. Custom PyTorch Dataset ---
class TextEmotionDataset(Dataset):
    """Custom Dataset class for PyTorch."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- 3. Model Architecture ---
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

# --- 4. Training Configuration ---
class TrainingConfig:
    """Configuration class for training hyperparameters."""
    def __init__(self):
        self.batch_size = 32  # Increased for faster training
        self.learning_rate = 3e-5  # Slightly higher for faster convergence
        self.num_epochs = 2   # Reduced to 2 epochs for quicker training
        self.warmup_steps = 50  # Reduced warmup steps
        self.max_grad_norm = 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Save in the same directory as the script
        self.save_path = './'
        self.max_samples = 2000  # Limit training samples for faster training

# --- 5. Training Function ---
def train_model(model, train_loader, val_loader, config):
    """The main training loop."""
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0.0

    # Create model save directory if it doesn't exist
    os.makedirs(config.save_path, exist_ok=True)

    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Training')

        for batch_idx, batch in enumerate(train_progress):
            try:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                
                total_train_loss += loss.item()
                train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        correct_predictions, total_predictions = 0, 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Validation')
            for batch in val_progress:
                try:
                    input_ids = batch['input_ids'].to(config.device)
                    attention_mask = batch['attention_mask'].to(config.device)
                    labels = batch['label'].to(config.device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    
                    val_progress.set_postfix({'accuracy': f'{correct_predictions/total_predictions:.4f}'})
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(config.save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, model_save_path)
            print(f"✓ New best model saved at {model_save_path} with accuracy: {best_val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies

# --- 6. Main Execution ---
def main_training():
    """Ties everything together to run the training process."""
    try:
        train_df, val_df, _ = load_text_data()
        
        config = TrainingConfig()
        print(f"Using device: {config.device}")

        print("Loading BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print("Creating datasets...")
        # Limit dataset size for faster training on CPU
        if len(train_df) > config.max_samples:
            print(f"Limiting training data to {config.max_samples} samples for faster training")
            train_df = train_df.sample(n=config.max_samples, random_state=42).reset_index(drop=True)
        
        if len(val_df) > 500:
            print(f"Limiting validation data to 500 samples for faster validation")
            val_df = val_df.sample(n=500, random_state=42).reset_index(drop=True)
            
        train_dataset = TextEmotionDataset(train_df['text'], train_df['label'], tokenizer)
        val_dataset = TextEmotionDataset(val_df['text'], val_df['label'], tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        print("Initializing model...")
        model = BertEmotionClassifier(n_classes=6)
        
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, config)

        # Plot training curves
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.savefig('training_curves.png')
            print("Training curves saved as 'training_curves.png'")
            plt.show()
        except Exception as e:
            print(f"Could not save/show plots: {e}")
            
        print(f"\n✓ Training completed! Best validation accuracy: {max(val_accuracies):.4f}")
        print("Model saved as 'best_model.pth'")
        print("You can now run 'python app.py' to start the web interface!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your internet connection and ensure all dependencies are installed:")
        print("pip install torch transformers datasets pandas numpy matplotlib scikit-learn tqdm gradio")

if __name__ == "__main__":
    main_training()