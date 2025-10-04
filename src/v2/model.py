"""
Neural language model architecture for text generation.
"""

import torch
import torch.nn as nn
import math


class NextWordPredictorLstm(nn.Module):
    """LSTM-based neural language model with attention mechanism for text generation."""

    def __init__(self, vocabulary_size, embedding_matrix, hidden_dimension, layer_count, padding_token_id):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_matrix.shape[1]
        self.hidden_dimension = hidden_dimension
        self.layer_count = layer_count
        self.padding_token_id = padding_token_id

        # Initialize embedding layer with pre-trained embeddings
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix).float(),
            freeze=False,
            padding_idx=self.padding_token_id
        )

        # LSTM layer for sequential processing
        self.sequence_processor = nn.LSTM(
            self.embedding_dimension,
            self.hidden_dimension,
            self.layer_count,
            batch_first=True
        )

        # Attention mechanism
        self.attention_layer = nn.Linear(self.hidden_dimension, 1)

        # Output projection layer
        self.output_projection = nn.Linear(self.hidden_dimension * 2, self.vocabulary_size)

        # Regularization
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, token_ids, hidden_state=None):
        """Forward pass through the neural language model."""
        # Apply word embeddings with dropout
        embedded_tokens = self.dropout_layer(self.word_embeddings(token_ids))

        # Process through LSTM
        sequence_output, updated_hidden = self.sequence_processor(embedded_tokens, hidden_state)

        # Apply attention mechanism
        attention_scores = self.attention_layer(sequence_output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute attended representation
        attended_representation = torch.bmm(attention_weights.transpose(1, 2), sequence_output)

        # Combine LSTM output with attended representation
        combined_features = torch.cat([sequence_output, attended_representation.expand_as(sequence_output)], dim=-1)

        # Project to vocabulary size
        logits = self.output_projection(combined_features)

        return logits, updated_hidden

    def initialize_hidden_state(self, batch_size, device):
        """Initialize hidden state for LSTM."""
        return (
            torch.zeros(self.layer_count, batch_size, self.hidden_dimension).to(device),
            torch.zeros(self.layer_count, batch_size, self.hidden_dimension).to(device)
        )


def prepare_training_sequences(token_sequences, sequence_length=40):
    """
    Create input/target sequence pairs for language model training.

    Args:
        token_sequences: List of token ID sequences
        sequence_length: Maximum length of input sequences

    Returns:
        Tuple of (input_tensors, target_tensors)
    """
    input_samples = []
    target_samples = []

    for token_sequence in token_sequences:
        if len(token_sequence) < sequence_length + 1:
            continue

        # Create sliding window sequences
        for start_idx in range(len(token_sequence) - sequence_length):
            input_sequence = token_sequence[start_idx:start_idx + sequence_length]
            target_sequence = token_sequence[start_idx + 1:start_idx + sequence_length + 1]
            input_samples.append(input_sequence)
            target_samples.append(target_sequence)

    input_tensor = torch.tensor(input_samples, dtype=torch.long)
    target_tensor = torch.tensor(target_samples, dtype=torch.long)
    return input_tensor, target_tensor


def evaluate_model_metrics(language_model, data_loader, computation_device):
    """
    Calculate both accuracy and perplexity metrics for the language model.

    Args:
        language_model: The neural language model
        data_loader: DataLoader with validation/test data
        computation_device: Device for computation (CPU/GPU)

    Returns:
        tuple: (accuracy, perplexity)
    """
    language_model.eval()

    # Metrics tracking
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0
    total_tokens = 0

    loss_function = nn.CrossEntropyLoss(ignore_index=language_model.padding_token_id, reduction='sum')

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(computation_device)
            target_batch = target_batch.to(computation_device)

            batch_size = input_batch.size(0)
            hidden_state = language_model.initialize_hidden_state(batch_size, computation_device)

            prediction_logits, _ = language_model(input_batch, hidden_state)

            # Calculate accuracy
            predicted_tokens = torch.argmax(prediction_logits, dim=-1)
            correct_predictions += (predicted_tokens == target_batch).sum().item()
            total_predictions += target_batch.numel()

            # Calculate loss for perplexity
            loss = loss_function(
                prediction_logits.view(-1, language_model.vocabulary_size),
                target_batch.view(-1)
            )
            total_loss += loss.item()
            total_tokens += target_batch.numel()

    # Calculate final metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    average_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(average_loss)

    return accuracy, perplexity


def evaluate_language_model(language_model, test_data_loader, computation_device):
    """
    Comprehensive evaluation of the language model on test data.

    Args:
        language_model: The neural language model to evaluate
        test_data_loader: DataLoader with test data
        computation_device: Device for computation (CPU/GPU)

    Returns:
        Tuple of (accuracy, perplexity)
    """
    print("Evaluating language model on test dataset...")
    test_accuracy, test_perplexity = evaluate_model_metrics(language_model, test_data_loader, computation_device)

    print(f"Final Test Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Perplexity: {test_perplexity:.2f}")

    return test_accuracy, test_perplexity



