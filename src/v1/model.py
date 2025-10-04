import torch
import torch.nn as nn


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


def load_model(model_path='best_sherlock_model.pth'):
    """
    Load a saved Sherlock model for inference
    """
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']
    seq_length = checkpoint['seq_length']
    
    model = LSTMWithAttention(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model loaded successfully!")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Sequence length: {seq_length}")
    
    return model, vocab_size, seq_length
