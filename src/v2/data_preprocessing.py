"""
Text preprocessing and embedding generation for neural language modeling.
"""

import re
import random
import numpy as np
from collections import Counter
import json
import os
import requests


def save_training_data(embeddings, vocab_data, sequences, base_path="data"):
    """Save training data in portable formats."""
    os.makedirs(base_path, exist_ok=True)

    # Save embeddings as NumPy
    np.save(f'{base_path}/embeddings.npy', embeddings)

    # Save vocabulary as JSON
    with open(f'{base_path}/vocabulary.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)

    # Save sequences as compressed NumPy
    np.savez_compressed(f'{base_path}/sequences.npz',
                       train=sequences['train'],
                       val=sequences['val'],
                       test=sequences['test'])

    print(f"Training data saved to {base_path}/ (embeddings.npy, vocabulary.json, sequences.npz)")


def load_training_data(base_path="data"):
    """Load training data from portable formats."""
    embeddings = np.load(f'{base_path}/embeddings.npy')

    with open(f'{base_path}/vocabulary.json', 'r') as f:
        vocab_data = json.load(f)

    sequences = np.load(f'{base_path}/sequences.npz')

    return embeddings, vocab_data, sequences


def check_training_data_exists(base_path="data"):
    """Check if all training data files exist."""
    required_files = [
        f'{base_path}/embeddings.npy',
        f'{base_path}/vocabulary.json',
        f'{base_path}/sequences.npz'
    ]
    return all(os.path.exists(f) for f in required_files)


def download_sherlock_holmes_text(file_path="data/sherlock_holmes.txt"):
    """Download Sherlock Holmes text from Project Gutenberg if not already present."""
    if os.path.exists(file_path):
        print(f"Text file already exists: {file_path}")
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    print(f"Downloading Sherlock Holmes text from Project Gutenberg...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Successfully downloaded: {file_path}")
        return file_path

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download text file: {e}")


def process_text_and_generate_embeddings(source_file, train_fraction=0.7, val_fraction=0.2):
    """
    Process raw text file and generate word embeddings for neural network training.

    Args:
        source_file: Path to input text file
        train_fraction: Proportion of data for training
        val_fraction: Proportion of data for validation

    Returns:
        Tuple of (train_sequences, val_sequences, test_sequences, token_to_id, id_to_token, embedding_matrix)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers library required: pip install sentence-transformers")

    # Load and preprocess raw text
    with open(source_file, "r", encoding="utf-8") as file_handle:
        raw_content = file_handle.read()

    # Extract main text content between Project Gutenberg markers
    content_parts = raw_content.split("*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK\nHOLMES ***")
    content_parts = content_parts[1].split("*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK\nHOLMES ***")

    # Clean and normalize text
    processed_text = content_parts[0]
    char_replacements = {
        "\r\n": " ", "\n": " ", "£": "pounds", "½": "one half", "&": "and",
        ":": "", "(": "", ")": "", "—": " ", "–": " ", ";": "", "_": " ",
        "à": "a", "â": "a", "æ": "ae", "è": "e", "é": "e", "œ": "oe",
        "'": "", "'": "", """: "", """: ""
    }

    for old_char, new_char in char_replacements.items():
        processed_text = processed_text.replace(old_char, new_char)

    processed_text = processed_text.strip()
    processed_text = re.sub(r'\s+', ' ', processed_text)
    sentence_segments = re.split(r'(?<=[.!?])\s+', processed_text)

    # Create text chunks for training
    text_chunks = []
    current_chunk = ""
    chunk_size_limit = 120

    for sentence in sentence_segments:
        current_chunk += " " + sentence
        if len(current_chunk) > chunk_size_limit:
            text_chunks.append(current_chunk.strip())
            current_chunk = sentence

    random.shuffle(text_chunks)
    total_chunks = len(text_chunks)
    train_end_idx = int(total_chunks * train_fraction)
    val_end_idx = int(total_chunks * (train_fraction + val_fraction))

    training_chunks = text_chunks[:train_end_idx]
    validation_chunks = text_chunks[train_end_idx:val_end_idx]
    testing_chunks = text_chunks[val_end_idx:]

    # Tokenize text chunks into words and symbols
    def tokenize_text(chunk_list):
        tokenized_chunks = []
        for text_chunk in chunk_list:
            normalized_chunk = text_chunk.lower()
            tokens = re.findall(r'\w+|[^\w\s]|\s+', normalized_chunk)
            tokenized_chunks.append(tokens)
        return tokenized_chunks

    train_tokens = tokenize_text(training_chunks)
    val_tokens = tokenize_text(validation_chunks)
    test_tokens = tokenize_text(testing_chunks)

    # Build vocabulary from all tokens
    token_frequency = Counter()
    for token_chunk in train_tokens + val_tokens + test_tokens:
        token_frequency.update(token_chunk)

    # Filter tokens by frequency and add special tokens
    min_frequency = 2
    vocabulary_tokens = [token for token, freq in token_frequency.items() if freq > min_frequency]
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<STOP>']
    complete_vocabulary = special_tokens + vocabulary_tokens

    # Create token-to-index and index-to-token mappings
    token_to_id = {token: idx for idx, token in enumerate(complete_vocabulary)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    # Generate embeddings using SentenceTransformers
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_vectors = []
    embedding_dimension = embedding_model.encode(['sample']).shape[1]

    for token in complete_vocabulary:
        if token in special_tokens:
            if token == '<PAD>':
                vector = np.zeros(embedding_dimension)
            else:  # <UNK>, <START>, <END>
                vector = np.random.normal(0, 0.1, embedding_dimension)
        else:
            vector = embedding_model.encode([token])[0]
        embedding_vectors.append(vector)

    embedding_matrix = np.vstack(embedding_vectors)

    # Convert tokenized chunks to sequences of indices
    def convert_tokens_to_indices(tokenized_chunks, token_mapping):
        index_sequences = []
        for token_chunk in tokenized_chunks:
            index_sequence = []
            for token in token_chunk:
                token_id = token_mapping.get(token, token_mapping['<UNK>'])
                index_sequence.append(token_id)
            index_sequences.append(index_sequence)
        return index_sequences

    train_sequences = convert_tokens_to_indices(train_tokens, token_to_id)
    val_sequences = convert_tokens_to_indices(val_tokens, token_to_id)
    test_sequences = convert_tokens_to_indices(test_tokens, token_to_id)

    return train_sequences, val_sequences, test_sequences, token_to_id, id_to_token, embedding_matrix


def setup_training_data(source_file="data/sherlock_holmes.txt", force_rebuild=False):
    """
    Main preprocessing function that creates and saves training data.

    Args:
        source_file: Path to input text file
        force_rebuild: Whether to recreate data even if files exist

    Returns:
        bool: True if processing was performed, False if existing files were used
    """
    # Download text file if it doesn't exist
    source_file = download_sherlock_holmes_text(source_file)

    if not force_rebuild and check_training_data_exists():
        print("Training data files already exist. Skipping preprocessing.")
        return False

    train_seqs, val_seqs, test_seqs, token_to_id, id_to_token, embed_matrix = process_text_and_generate_embeddings(source_file)

    sequences = {
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs
    }

    vocab_data = {
        'word_to_idx': token_to_id,
        'idx_to_word': id_to_token
    }

    save_training_data(embed_matrix, vocab_data, sequences)
    return True


if __name__ == "__main__":
    setup_training_data()
