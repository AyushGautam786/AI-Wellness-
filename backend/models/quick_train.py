import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
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
        dataset = load_dataset("emotion")
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    
    # QUICK TRAINING: Use only small subset
    print("Using QUICK TRAINING mode - small dataset for fast testing")
    train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(n=100, random_state=42).reset_index(drop=True)

    # Map labels to human-readable names
    emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    train_df['emotion_name'] = train_df['label'].map(emotion_mapping)
    val_df['emotion_name'] = val_df['label'].map(emotion_mapping)
    
    print(f"Quick training - Train: {len(train_df)}, Validation: {len(val_df)}")
    return train_df, val_df

# --- 2. Custom PyTorch Dataset ---
class TextEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):  # Reduced max_length
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

# --- 3. Lightweight Model Architecture ---
class SimpleBertClassifier(nn.Module):
    """Simplified BERT classifier for quick training."""
    def __init__(self, n_classes, model_name='distilbert-base-uncased', dropout=0.3):
        super(SimpleBertClassifier, self).__init__()
        # Use DistilBERT (smaller, faster version of BERT)
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
        output = self.dropout(pooled_output)
        return self.classifier(output)

# --- 4. Quick Training Configuration ---
class QuickTrainingConfig:
    def __init__(self):
        self.batch_size = 64  # Larger batch size
        self.learning_rate = 5e-5  # Higher learning rate
        self.num_epochs = 1  # Just 1 epoch for quick test
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = './'

# --- 5. Quick Training Function ---
def quick_train_model(model, train_loader, val_loader, config):
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Quick training for {config.num_epochs} epoch on {config.device}")
    print(f"Batch size: {config.batch_size}")

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        
        # Training
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc='Training')

        for batch in train_progress:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_predictions
        print(f'Validation Accuracy: {accuracy:.4f}')

        # Save model
        model_save_path = os.path.join(config.save_path, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': accuracy,
        }, model_save_path)
        print(f"✓ Model saved at {model_save_path}")

# --- 6. Main Execution ---
def main():
    try:
        train_df, val_df = load_text_data()
        
        config = QuickTrainingConfig()
        print(f"Using device: {config.device}")

        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        train_dataset = TextEmotionDataset(train_df['text'], train_df['label'], tokenizer)
        val_dataset = TextEmotionDataset(val_df['text'], val_df['label'], tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        print("Initializing lightweight model...")
        model = SimpleBertClassifier(n_classes=6)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model has {param_count:,} parameters (much smaller than full BERT!)")

        quick_train_model(model, train_loader, val_loader, config)
        
        print(f"\n✅ Quick training completed!")
        print("🚀 You can now run 'python app.py' to test the model!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()