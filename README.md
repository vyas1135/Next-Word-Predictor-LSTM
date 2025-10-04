# Next Word Predictor

A next word prediction tool built with LSTM and Attention mechanism. This project trains a neural network to predict the next word in a sentence using Project Gutenberg Ebook "The Adventures of Sherlock Holmes" as the training corpus.

## Overview

This project includes:

- **data_preprocessing.py**: Downloads, cleans, tokenizes, and prepares embeddings from the text
- **model.py**: Defines the LSTM+Attention neural network architecture
- **training.py**: Complete training pipeline with evaluation metrics
- **text_generator.py**: Word by word text generation with top-p sampling

## Architecture

The model consists of:

- **Embedding Layer**: Converts tokenized words to dense vector embeddings using pre-trained word embeddings
- **LSTM Layer**: 256 hidden units and 1 layer for learning sequential patterns
- **Attention Layer**:  Helps focus on the most relevant parts of the input
- **Dense Output Layer**: Predicts probability scores over entire vocabulary

## 📁 Project Structure

```
Next-Word-Prediction-LSTM/
├── src/v2/
│   ├── model.py              # Neural network architecture
│   ├── data_preprocessing.py # Data processing and embedding generation
│   ├── training.py           # Complete training pipeline
│   └── text_generator.py     # Text generation and inference
├── pyproject.toml           # Dependencies
└── README.md                # This file
```

## How to Run

### 1. Clone Repository and move to project directory
```bash
git clone https://github.com/vyas1135/Next-Word-Predictor-LSTM.git
cd Next-Word-Prediction-LSTM
```

### 2. Install Dependencies
Ensure Poetry is installed, then run:
```bash
poetry install
```

### 3. Run Complete Pipeline (One Command)
```bash
poetry run next-word-predictor
```

**What it does:**
- Downloads the text file (if not available)
- Processes the text into training data with embeddings
- Trains the LSTM+Attention model for 150 epochs and evaluates on test data
- Saves the best model checkpoint
- Generates sample text with prediction details
- Shows training results, accuracy, perplexity, and example outputs


## Model Performance

**Training Configuration:**
- Sequence Length: 40 tokens
- Hidden Dimension: 256
- Training Epochs: 150
- Batch Size: 256
- Model Parameters: 4,595,495

**Performance Metrics:**
- Training Accuracy: 91.85%
- Test Accuracy: 80.43%
- Perplexity: 3.02

## Sample Generated Text

**Example 1:**
```
Prompt: I asked
Generated text: i asked upon him. “perhaps it looks against me how well?’ “yes, mr-dozen, but what about them
```

**Example 2:**
```
Prompt: every precaution has
Generated text: every precaution has to explain some terrible mysterious subject because he should vanish little for the purpose of our art in great life,
```
