#!/usr/bin/env python3
"""Sinhala-specific vocabulary extension for XTTS"""

import json
import os
import argparse
import pandas as pd
from collections import Counter
from sinling import SinhalaTokenizer

def create_sinhala_vocab(metadata_path, output_path, vocab_size=2500):
    """Create BPE vocabulary specifically for Sinhala using sinling tokenizer"""

    # Initialize the tokenizer
    tokenizer = SinhalaTokenizer()
    
    # Read metadata
    df = pd.read_csv(metadata_path, sep="|", header=None,
                     names=["audio_file", "text", "speaker"])
    texts = df.text.tolist()
    
    print(f"Processing {len(texts)} Sinhala texts with sinling tokenizer...")
    
    # Tokenize all texts and create a flat list of tokens
    all_tokens = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)
        
    # Calculate token frequencies
    token_freq = Counter(all_tokens)

    print(f"Found {len(token_freq)} unique Sinhala tokens.")
    
    # Build vocabulary
    vocab = {}
    idx = 0
    
    # Special tokens
    special_tokens = ["[PAD]", "[UNK]", "[START]", "[STOP]", "[si]"]
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # Add most common tokens from the dataset
    # Sort tokens by frequency in descending order
    most_common_tokens = token_freq.most_common(vocab_size - len(special_tokens))
    
    for token, freq in most_common_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    print(f"✅ Final vocabulary size: {len(vocab)} tokens")
    
    # Save vocabulary
    vocab_dir = os.path.join(output_path, "XTTS-v2")
    os.makedirs(vocab_dir, exist_ok=True)
    
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved vocabulary to: {vocab_path}")
    
    # Statistics
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   - Special tokens: {len(special_tokens)}")
    print(f"   - Sinhala tokens from sinling: {len(most_common_tokens)}")
    print(f"   - Total: {len(vocab)} tokens")
    
    return vocab_path

def adjust_config(args):
    """Update config.json with Sinhala language"""
    config_path = os.path.join(args.output_path, "XTTS-v2/config.json")
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
        
        if "languages" not in config:
            config["languages"] = []
        
        # Add 'si' if it's not already in the list
        if "si" not in config["languages"]:
            config["languages"].append("si")
        
        # Update the number of text tokens to match the new vocab size
        if "model_args" in config:
            config["model_args"]["gpt_num_text_tokens"] = len(json.load(open(os.path.join(args.output_path, "XTTS-v2/vocab.json"))))
            config["model_args"]["gpt_start_text_token"] = 2  # Assuming [START] is at index 2
            config["model_args"]["gpt_stop_text_token"] = 3   # Assuming [STOP] is at index 3
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Updated config.json with language 'si' and new vocab size.")
    except Exception as e:
        print(f"⚠️ Could not update config.json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Sinhala vocabulary for XTTS using sinling tokenizer.")
    
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to the training metadata CSV file.")
    parser.add_argument("--output_path", type=str, default="checkpoints/",
                       help="Directory to save the output vocab.json and config.json.")
    parser.add_argument("--vocab_size", type=int, default=2500,
                       help="The target size of the vocabulary.")
    
    args = parser.parse_args()
    
    vocab_path = create_sinhala_vocab(
        args.metadata_path,
        args.output_path,
        args.vocab_size
    )
    
    adjust_config(args)
    
    print(f"\n✅ Sinhala tokenization complete!")
    print(f"   A new vocab.json and updated config.json have been saved in {args.output_path}/XTTS-v2/")
