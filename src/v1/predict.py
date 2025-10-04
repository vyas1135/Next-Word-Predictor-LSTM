import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
from v1.model import load_model

def generate_text(seed_text, max_phrase_len, model=None):
    """
    Generates text by predicting the next word in a loop.
    Also shows the top 5 predictions at each step as required.
    """
    # Validate inputs
    if not seed_text or not isinstance(seed_text, str):
        raise ValueError("seed_text must be a non-empty string")
    if max_phrase_len <= 0:
        raise ValueError("max_phrase_len must be positive")
    
    # Load model if not provided
    if model is None:
        model, _, _ = load_model()
    
    model.eval()
    
    # Load tokenizer (try both possible names)
    sp = spm.SentencePieceProcessor()
    if os.path.exists('sherlock.model'):
        sp.load('sherlock.model')
    elif os.path.exists('sherlock_sp_2.model'):
        sp.load('sherlock_sp_2.model')
    else:
        raise FileNotFoundError("Tokenizer model not found")
    
    # Encode seed text - handle empty/short prompts
    tokens = sp.encode_as_ids(seed_text.strip())
    if not tokens:  # Handle edge case of empty tokenization
        tokens = [1]  # Start with a basic token
    
    generated = tokens[-59:] if len(tokens) > 59 else tokens
    
    words_and_probabilities = []
    
    with torch.no_grad():
        for step in range(max_phrase_len):
            # Pad to sequence length (48)
            input_seq = [0] * (48 - len(generated)) + generated if len(generated) < 48 else generated[-48:]
            x = torch.tensor([input_seq])
            
            logits = model(x)[0]
            
            # Get top-5 predictions
            top5_logits, top5_idx = torch.topk(logits, 5)
            top5_probs = F.softmax(top5_logits, dim=-1)
            
            # Convert to words and probabilities
            top5_words = []
            for idx, prob in zip(top5_idx, top5_probs):
                word = sp.decode_ids([idx.item()])
                top5_words.append((word, prob.item()))
            
            # Sample next token (using top-k for better quality)
            top_k_logits, top_k_indices = torch.topk(logits, 25)
            top_k_probs = F.softmax(top_k_logits / 0.7, dim=-1)
            next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
            
            generated.append(next_token)
            
            # Store step results
            chosen_word = sp.decode_ids([next_token])
            words_and_probabilities.append({
                'step': step + 1,
                'chosen': chosen_word,
                'top5': top5_words
            })
            
            # Stop on special tokens
            if next_token in [0, 1, 2, 3]:
                break
    
    # Generate final output text
    output_text = sp.decode_ids(generated)
    
    return output_text, words_and_probabilities

def demo():
    """Demo function for evaluator to test"""
    print("SHERLOCK HOLMES NEXT WORD PREDICTION")
    print("="*50)
    
    # Test prompts
    prompts = ["Holmes said", "Watson observed", "The door opened"]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result, steps = generate_text(prompt, max_phrase_len=5)
        print(f"Generated: {result}")
        print("Top-5 predictions:")
        for step in steps[:3]:
            print(f"  Step {step['step']}: '{step['chosen']}'")
            for j, (word, prob) in enumerate(step['top5'][:3]):
                print(f"    {j+1}. '{word}' ({prob:.3f})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # python predict.py "prompt" num_words
        prompt = sys.argv[1]
        try:
            num_words = int(sys.argv[2])
        except (ValueError, IndexError):
            num_words = 10
        result, steps = generate_text(prompt, max_phrase_len=num_words)
        print(f"Prompt: '{prompt}'")
        print(f"Generated ({num_words} tokens): {result}")
    elif len(sys.argv) > 1:
        # python predict.py "prompt" (default 30 words)
        prompt = sys.argv[1]
        result, steps = generate_text(prompt, max_phrase_len=30)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {result}")
    else:
        # Interactive mode
        print("SHERLOCK HOLMES NEXT WORD PREDICTION")
        print("Usage: python predict.py \"prompt\" [num_tokens]")
        print("Or run interactively:")
        
        prompt = input("Enter prompt: ").strip()
        if prompt:
            try:
                num_tokens = int(input("Number of tokens to generate (default 30): ") or "30")
            except ValueError:
                num_tokens = 10
            
            result, steps = generate_text(prompt, max_phrase_len=num_tokens)
            print(f"\nGenerated ({num_tokens} tokens): {result}")
