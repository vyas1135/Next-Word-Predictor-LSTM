"""
Neural language model training system with advanced optimization techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import (NextWordPredictorLstm, prepare_training_sequences,
                   evaluate_model_metrics, evaluate_language_model)
from data_preprocessing import load_training_data


def train_neural_language_model(language_model, training_data_loader, validation_data_loader,
                                training_epochs, vocabulary_size, computation_device="cpu",
                                gradient_clip_threshold=1.0, checkpoint_save_path="best_model.pth"):
    """
    Train neural language model with advanced optimization and monitoring.

    Args:
        language_model: Neural language model to train
        training_data_loader: DataLoader for training data
        validation_data_loader: DataLoader for validation data
        training_epochs: Number of training epochs
        vocabulary_size: Size of the vocabulary
        computation_device: Device for computation (CPU/GPU)
        gradient_clip_threshold: Gradient clipping threshold
        checkpoint_save_path: Path to save best model checkpoint

    Returns:
        Tuple of (best_validation_loss, best_epoch_number)
    """
    language_model.train()
    print("Initiating neural language model training...")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(language_model.parameters(), lr=0.00001)
    learning_rate_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,
        steps_per_epoch=len(training_data_loader),
        epochs=training_epochs,
        pct_start=0.1
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=language_model.padding_token_id)

    # Training monitoring variables
    best_validation_loss = float('inf')
    best_epoch_number = 0
    early_stopping_patience = 7
    patience_counter = 0

    for current_epoch in range(training_epochs):
        epoch_total_loss = 0

        # Training progress tracking
        training_progress_bar = tqdm(
            training_data_loader,
            desc=f"Epoch {current_epoch+1}/{training_epochs} [Training]",
            leave=False, dynamic_ncols=True
        )

        for batch_index, (input_batch, target_batch) in enumerate(training_progress_bar):
            # Move data to computation device
            input_batch = input_batch.to(computation_device)
            target_batch = target_batch.to(computation_device)

            # Initialize hidden state
            batch_size = input_batch.size(0)
            hidden_state = language_model.initialize_hidden_state(batch_size, computation_device)

            # Forward pass
            optimizer.zero_grad()
            prediction_logits, hidden_state = language_model(input_batch, hidden_state)

            # Detach hidden state to prevent backprop through time
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

            # Calculate loss
            batch_loss = loss_function(
                prediction_logits.view(-1, vocabulary_size),
                target_batch.view(-1)
            )
            epoch_total_loss += batch_loss.item()

            # Backward pass and optimization
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(language_model.parameters(), gradient_clip_threshold)
            optimizer.step()

            # Update training progress display
            training_progress_bar.set_postfix({
                'batch_loss': f'{batch_loss.item():.4f}',
                'avg_loss': f'{epoch_total_loss/(batch_index+1):.4f}'
            })

        # Calculate average training loss for epoch
        average_training_loss = epoch_total_loss / len(training_data_loader)

        # Validation phase with progress tracking
        language_model.eval()
        validation_total_loss = 0
        validation_progress_bar = tqdm(
            validation_data_loader,
            desc=f"Epoch {current_epoch+1}/{training_epochs} [Validation]",
            leave=False, dynamic_ncols=True
        )

        with torch.no_grad():
            for val_batch_index, (val_input_batch, val_target_batch) in enumerate(validation_progress_bar):
                # Move validation data to device
                val_input_batch = val_input_batch.to(computation_device)
                val_target_batch = val_target_batch.to(computation_device)

                # Initialize validation hidden state
                val_batch_size = val_input_batch.size(0)
                val_hidden_state = language_model.initialize_hidden_state(val_batch_size, computation_device)

                # Forward pass for validation
                val_prediction_logits, val_hidden_state = language_model(val_input_batch, val_hidden_state)
                val_hidden_state = (val_hidden_state[0].detach(), val_hidden_state[1].detach())

                # Calculate validation loss
                val_batch_loss = loss_function(
                    val_prediction_logits.view(-1, vocabulary_size),
                    val_target_batch.view(-1)
                )
                validation_total_loss += val_batch_loss.item()

                # Update validation progress display
                validation_progress_bar.set_postfix({
                    'val_batch_loss': f'{val_batch_loss.item():.4f}',
                    'avg_val_loss': f'{validation_total_loss/(val_batch_index+1):.4f}'
                })

        # Calculate average validation loss
        average_validation_loss = validation_total_loss / len(validation_data_loader)
        learning_rate_scheduler.step()

        # Model checkpoint management
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_epoch_number = current_epoch + 1
            patience_counter = 0

            # Create model checkpoint
            model_checkpoint = {
                'epoch': current_epoch + 1,
                'model_state_dict': language_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': learning_rate_scheduler.state_dict(),
                'train_loss': average_training_loss,
                'val_loss': average_validation_loss,
                'vocab_size': vocabulary_size,
                'hidden_dim': language_model.hidden_dimension,
                'num_layers': language_model.layer_count,
                'pad_token_id': language_model.padding_token_id
            }
            torch.save(model_checkpoint, checkpoint_save_path)
            print(f"New best model checkpoint saved! Validation Loss: {average_validation_loss:.4f}")
        else:
            patience_counter += 1

        # Comprehensive metrics evaluation every 10 epochs
        if (current_epoch + 1) % 10 == 0:
            print("Computing comprehensive evaluation metrics...")
            training_accuracy, training_perplexity = evaluate_model_metrics(language_model, training_data_loader, computation_device)
            validation_accuracy, validation_perplexity = evaluate_model_metrics(language_model, validation_data_loader, computation_device)

            print(f"Epoch [{current_epoch+1}/{training_epochs}] Comprehensive Results:")
            print(f"  Training Loss: {average_training_loss:.4f}, Validation Loss: {average_validation_loss:.4f}")
            print(f"  Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
            print(f"  Training Perplexity: {training_perplexity:.2f}, Validation Perplexity: {validation_perplexity:.2f}")
            print(f"  Best Validation Loss: {best_validation_loss:.4f} (Epoch {best_epoch_number})")
            print(f"  Early Stopping Patience: {patience_counter}/{early_stopping_patience}")
            print("-" * 80)
        else:
            print(f"Epoch [{current_epoch+1}/{training_epochs}] Training Loss: {average_training_loss:.4f}, "
                  f"Validation Loss: {average_validation_loss:.4f} | Best: {best_validation_loss:.4f}")

        # Early stopping mechanism
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping activated! No improvement for {early_stopping_patience} epochs.")
            break

        # Return to training mode for next epoch
        language_model.train()

    print(f"Training completed successfully! Best model from epoch {best_epoch_number} "
          f"with validation loss {best_validation_loss:.4f}")
    return best_validation_loss, best_epoch_number


def execute_complete_training_pipeline():
    """
    Execute complete neural language model training pipeline.
    Loads data, trains model, and evaluates performance.
    """
    # Load preprocessed data and embeddings
    embedding_matrix, vocab_data, sequences = load_training_data()

    # Extract training data
    training_sequences = sequences['train']
    validation_sequences = sequences['val']
    testing_sequences = sequences['test']

    # Extract vocabulary mappings
    token_to_id_mapping = vocab_data['word_to_idx']

    # Initialize training parameters
    padding_token_id = token_to_id_mapping["<PAD>"]
    vocabulary_size = len(token_to_id_mapping)
    training_batch_size = 256
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computation device: {computation_device}")

    # Prepare data loaders for training, validation, and testing
    print("Creating data loaders for training pipeline...")

    # Data loading
    train_inputs, train_targets = prepare_training_sequences(training_sequences)
    training_dataset = TensorDataset(train_inputs, train_targets)
    training_data_loader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True)

    
    val_inputs, val_targets = prepare_training_sequences(validation_sequences)
    validation_dataset = TensorDataset(val_inputs, val_targets)
    validation_data_loader = DataLoader(validation_dataset, batch_size=training_batch_size, shuffle=False)

    
    test_inputs, test_targets = prepare_training_sequences(testing_sequences)
    testing_dataset = TensorDataset(test_inputs, test_targets)
    testing_data_loader = DataLoader(testing_dataset, batch_size=training_batch_size, shuffle=False)

    # Initialize neural language model
    print("Initializing neural language model architecture...")
    language_model = NextWordPredictorLstm(
        vocabulary_size=vocabulary_size,
        embedding_matrix=embedding_matrix,
        hidden_dimension=256,
        layer_count=1,
        padding_token_id=padding_token_id
    )
    language_model.to(computation_device)
    total_parameters = sum(param.numel() for param in language_model.parameters())
    print(f"Total model parameters: {total_parameters:,}")

    # Execute training with advanced optimization
    train_neural_language_model(
        language_model, training_data_loader, validation_data_loader,
        training_epochs=150, vocabulary_size=vocabulary_size,
        computation_device=computation_device,
        checkpoint_save_path="best_sherlock_model.pth"
    )

    # Load and evaluate best trained model
    print("Loading best model checkpoint for final evaluation...")
    model_checkpoint = torch.load("best_sherlock_model.pth", map_location=computation_device)

    # Reconstruct best model
    best_language_model = NextWordPredictorLstm(
        vocabulary_size=model_checkpoint['vocab_size'],
        embedding_matrix=embedding_matrix,
        hidden_dimension=model_checkpoint['hidden_dim'],
        layer_count=model_checkpoint['num_layers'],
        padding_token_id=model_checkpoint['pad_token_id']
    )
    best_language_model.load_state_dict(model_checkpoint['model_state_dict'])
    best_language_model.to(computation_device)

    # Comprehensive model evaluation
    final_accuracy, final_perplexity = evaluate_language_model(best_language_model, testing_data_loader, computation_device)

    print("\nNeural language model training pipeline completed successfully!")

    # Generate sample text demonstrations
    print("\n" + "="*80)
    print("GENERATING SAMPLE TEXT DEMONSTRATIONS")
    print("="*80)

    from text_generator import generate_text

    # Prepare mappings for text generation
    id_to_token_mapping = {v: k for k, v in token_to_id_mapping.items()}

    # Sample prompts for demonstration
    sample_prompts = [
        "I saw Holmes",
        "The detective examined",
        "Watson observed that",
        "It was a dark",
        "The mystery of"
    ]

    print(f"\nFINAL MODEL PERFORMANCE:")
    print(f"  Test Accuracy: {final_accuracy:.4f}")
    print(f"  Test Perplexity: {final_perplexity:.2f}")
    print(f"  Model Parameters: {sum(p.numel() for p in best_language_model.parameters()):,}")

    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Generating from: '{prompt}'")

        try:
            generated_text = generate_text(
                best_language_model,
                token_to_id_mapping,
                id_to_token_mapping,
                prompt_text=prompt,
                generation_length=15,
                top_p_threshold=0.9,
                sampling_temperature=0.9,
                max_context_length=40,
                computation_device=computation_device
            )
            print(f"Generated text: {generated_text}")

        except Exception as e:
            print(f"Generation failed: {e}")

    # Show detailed prediction for the last prompt
    print(f"\n--- Detailed Prediction Analysis for: '{sample_prompts[-1]}' ---")
    try:
        detailed_text = generate_text(
            best_language_model,
            token_to_id_mapping,
            id_to_token_mapping,
            prompt_text=sample_prompts[-1],
            generation_length=10,
            top_p_threshold=0.9,
            sampling_temperature=0.8,
            max_context_length=40,
            computation_device=computation_device,
            show_predictions=True  # This will be added to generator
        )
        print(f"Generated: {detailed_text}")
    except Exception as e:
        print(f"Detailed generation failed: {e}")

    print("\nComplete pipeline finished! Model ready for use.")
    print("To generate more text, run: python generator.py")


if __name__ == "__main__":
    execute_complete_training_pipeline()
