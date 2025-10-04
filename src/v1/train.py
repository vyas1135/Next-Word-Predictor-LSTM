import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import sentencepiece as spm
import requests
import os
import numpy as np
from tqdm import tqdm
import math

def calculate_accuracy(outputs, targets, k):
    """Calculate top-k accuracy"""
    _, top_k_pred = torch.topk(outputs, k, dim=1)
    correct = top_k_pred.eq(targets.view(-1, 1).expand_as(top_k_pred))
    return correct.sum().float() / targets.size(0)

def evaluate_model(model, data_loader):
    """Evaluate model and return loss + accuracies"""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            total_loss += loss.item()
            
            all_outputs.append(output)
            all_targets.append(batch_y)
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate top-1 and top-5 accuracies
    accuracies = {}
    for k in [1, 5]:
        accuracies[f'top_{k}'] = calculate_accuracy(all_outputs, all_targets, k).item()
    
    avg_loss = total_loss / len(data_loader)
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity, accuracies

# Download data
def download_data():
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    if not os.path.exists("sherlock.txt"):
        print("Downloading Sherlock Holmes text...")
        response = requests.get(url)
        with open("sherlock.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open("sherlock.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple cleaning
    start = text.find("THE ADVENTURES OF SHERLOCK HOLMES")
    end = text.find("End of the Project Gutenberg EBook")
    if start != -1 and end != -1:
        text = text[start:end]
    
    return text

class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # Simple components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Simple attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Embed and LSTM
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)  # Residual connection
        
        # Take last token for prediction
        final_hidden = attn_out[:, -1, :]
        output = self.fc(final_hidden)
        
        return output

def train_model():
    # 1. Get data
    text = download_data()
    print(f"Text length: {len(text):,} characters")
    
    # 2. Train tokenizer with dynamic vocab size
    print("Training tokenizer...")
    with open("temp.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # Calculate dynamic vocab size based on text
    unique_words = len(set(text.split()))
    dynamic_vocab_size = 14100
    print(f"Dynamic vocab size: {dynamic_vocab_size} (based on {unique_words} unique words)")
    
    spm.SentencePieceTrainer.train(
        input="temp.txt",
        model_prefix="sherlock",
        vocab_size=dynamic_vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        split_by_whitespace=True
    )
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("sherlock.model")
    vocab_size = sp.get_piece_size()
    print(f"Final vocab size: {vocab_size}")
    
    # 3. Prepare training data
    tokens = sp.encode_as_ids(text)
    seq_len = 48
    
    X, y = [], []
    for i in range(len(tokens) - seq_len):
        X.append(tokens[i:i + seq_len])
        y.append(tokens[i + seq_len])
    
    X = torch.tensor(X)
    y = torch.tensor(y)
    
    # Split into train/validation
    dataset = TensorDataset(X, y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 4. Train model with improvements
    model = LSTMWithAttention(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=1
    )
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    print("Training model...")
    for epoch in range(10):
        # Training phase
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate training and validation
        train_loss, train_perplexity, train_acc = evaluate_model(model, train_loader)
        val_loss, val_perplexity, val_acc = evaluate_model(model, val_loader)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        print(f"  Train Acc - Top-1: {train_acc['top_1']:.3f}, Top-5: {train_acc['top_5']:.3f}")
        print(f"  Val Acc   - Top-1: {val_acc['top_1']:.3f}, Top-5: {val_acc['top_5']:.3f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'seq_length': seq_len,
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_sherlock_model.pth')
            print(f" New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training completed!")
    return model, vocab_size, seq_len

if __name__ == "__main__":
    train_model()
