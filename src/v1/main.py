"""
Complete Sherlock Holmes Text Generator
Run this for full training + prediction pipeline
"""
from v1.train import train_model
from v1.predict import generate_text, demo

def main():
    """Full pipeline: train model then generate text"""
    print("Starting full Sherlock Holmes generator pipeline...")
    
    # 1. Train the model
    print("\n" + "="*50)
    print("PHASE 1: TRAINING MODEL")
    print("="*50)
    model, vocab_size, seq_len = train_model()
    
    # 2. Test generation
    print("\n" + "="*50)
    print("PHASE 2: TESTING GENERATION")
    print("="*50)
    demo()

if __name__ == "__main__":
    main()
