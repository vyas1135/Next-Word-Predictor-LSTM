import torch
import pickle
import re
import os
from v1.model import NextWordPredictorLstm


def convert_text_to_token_ids(input_text, token_mapping, max_length=40):
    tokens = re.findall(r'\w+|[^\w\s]|\s+', input_text.lower()) # Include punctuation and whitespaces
    token_ids = [token_mapping.get(token, token_mapping['<UNK>']) for token in tokens]

    if max_length and len(token_ids) > max_length:
        token_ids = token_ids[-max_length:]

    return token_ids


def convert_token_ids_to_text(token_ids, id_mapping):
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    tokens = [
        id_mapping.get(token_id, '<UNK>')
        for token_id in token_ids
        if id_mapping.get(token_id, '<UNK>') not in special_tokens
    ]

    text = ' '.join(tokens)
    text = re.sub(r'\s([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentence_parts = re.split(r'([.!?])', text)
    formatted_parts = [part.capitalize() for part in sentence_parts if part.strip()]
    text = ''.join(formatted_parts)

    return text


def load_trained_model_and_embeddings(model_checkpoint_path, embeddings_file_path, computation_device="cpu"):
    with open(embeddings_file_path, 'rb') as embedding_file:
        embedding_data = pickle.load(embedding_file)

    token_to_id_mapping = embedding_data['word_to_idx']
    id_to_token_mapping = embedding_data['idx_to_word']
    embedding_matrix = embedding_data['embeddings_matrix']
    padding_token_id = token_to_id_mapping["<PAD>"]

    model_checkpoint = torch.load(model_checkpoint_path, map_location=computation_device)

    language_model = NextWordPredictorLstm(
        vocabulary_size=model_checkpoint['vocab_size'],
        embedding_matrix=embedding_matrix,
        hidden_dimension=model_checkpoint['hidden_dim'],
        layer_count=model_checkpoint['num_layers'],
        padding_token_id=model_checkpoint['pad_token_id']
    )

    language_model.load_state_dict(model_checkpoint['model_state_dict'])
    language_model.to(computation_device)
    language_model.eval()

    print(f" Model loaded from epoch {model_checkpoint['epoch']} | Loss: {model_checkpoint['val_loss']:.4f}")
    return language_model, token_to_id_mapping, id_to_token_mapping, padding_token_id


def generate_text(language_model, token_to_id_mapping, id_to_token_mapping,
                          prompt_text, generation_length=30, top_p_threshold=0.9,
                          sampling_temperature=0.9, max_context_length=40, computation_device="cpu"):
    language_model.eval()
    generated_token_ids = convert_text_to_token_ids(prompt_text, token_to_id_mapping, max_context_length)
    context_ids = generated_token_ids.copy()
    hidden_state = language_model.initialize_hidden_state(1, computation_device)
    repetition_prevention_window = 5

    for generation_step in range(generation_length):
        context_tensor = torch.tensor([context_ids], dtype=torch.long).to(computation_device)

        with torch.no_grad():
            prediction_logits, hidden_state = language_model(context_tensor, hidden_state)
            current_logits = prediction_logits[0, -1, :]
            token_probabilities = torch.softmax(current_logits / sampling_temperature, dim=-1)

            sorted_probabilities, sorted_token_indices = torch.sort(token_probabilities, descending=True)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=0)
            cutoff_index = torch.nonzero(cumulative_probabilities > top_p_threshold, as_tuple=False)

            if cutoff_index.numel() > 0:
                sorted_probabilities = sorted_probabilities[:cutoff_index[0].item()+1]
                sorted_token_indices = sorted_token_indices[:cutoff_index[0].item()+1]
                sorted_probabilities /= sorted_probabilities.sum()

            next_token_id = sorted_token_indices[torch.multinomial(sorted_probabilities, 1).item()].item()

            recent_token_ids = generated_token_ids[-repetition_prevention_window:]
            repetition_attempts = 0
            while next_token_id in recent_token_ids and repetition_attempts < 5:
                next_token_id = sorted_token_indices[torch.multinomial(sorted_probabilities, 1).item()].item()
                repetition_attempts += 1
            generated_token_ids.append(next_token_id)
            context_ids.append(next_token_id)

            # Maintain context window size
            if len(context_ids) > max_context_length:
                context_ids = context_ids[1:]

            # Display top predictions for debugging
            top_k_probabilities, top_k_indices = torch.topk(token_probabilities, 5)
            top_predictions = [
                (id_to_token_mapping[idx.item()], f"{prob.item():.3f}")
                for idx, prob in zip(top_k_indices, top_k_probabilities)
            ]
            print(f"Generation Step {generation_step+1} | Top 5 predictions: {top_predictions}")

    return convert_token_ids_to_text(generated_token_ids, id_to_token_mapping)


def main():
    """Main execution function for text generation demonstration."""
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using computation device: {computation_device}")

    
    embeddings_file = "embeddings.pkl"
    model_checkpoint_file = "best_sherlock_model.pth"

    if not os.path.exists(embeddings_file) or not os.path.exists(model_checkpoint_file):
        print(" Required model files not found. Please check the file paths.")
        return

    # Load trained model and embeddings
    language_model, token_to_id_mapping, id_to_token_mapping, _ = load_trained_model_and_embeddings(
        model_checkpoint_file, embeddings_file, computation_device
    )

    # Test prompts for generation
    test_prompts = [
        "I saw Holmes"
    ]

    for prompt in test_prompts:
        print(f"\n--- Starting Prompt: '{prompt}' ---")
        generated_text = generate_text(
            language_model, token_to_id_mapping, id_to_token_mapping,
            prompt_text=prompt,
            generation_length=30,
            top_p_threshold=0.9,
            sampling_temperature=1.0,
            max_context_length=40,
            computation_device=computation_device
        )
        print(f"Generated text: {generated_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()
