#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import re

def extract_sinhala_vocab(metadata_path, vocab_size=2500):
    """Extract unique Sinhala characters and BPE pairs from metadata."""
    
    characters = set()
    texts = []
    
    # Read metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 2:
                text = parts  # Second field is text
                texts.append(text)
                # Extract all unique characters
                for char in text:
                    characters.add(char)
    
    # Start with character-level tokens
    vocab = {char: idx for idx, char in enumerate(sorted(characters))}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    vocab["[START]"] = len(vocab)
    vocab["[STOP]"] = len(vocab)
    vocab["[SPACE]"] = len(vocab)
    
    print(f"Character-level vocabulary size: {len(vocab)}")
    
    # Simple BPE: iteratively merge frequent pairs
    texts_str = ' '.join(texts)
    pair_freq = defaultdict(int)
    current_vocab_size = len(vocab)
    
    # Build BPE tokens up to vocab_size
    iteration = 0
    while current_vocab_size < vocab_size and iteration < 1000:
        # Count adjacent character pairs
        pair_freq.clear()
        tokens = list(texts_str)
        
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i+1])
            pair_freq[pair] += 1
            i += 1
        
        if not pair_freq:
            break
        
        # Get most frequent pair
        best_pair = max(pair_freq, key=pair_freq.get)
        new_token = best_pair + best_pair
        
        # Add to vocabulary
        vocab[new_token] = current_vocab_size
        current_vocab_size += 1
        
        # Merge in text
        texts_str = texts_str.replace(best_pair + best_pair, new_token)
        
        iteration += 1
        if iteration % 100 == 0:
            print(f"BPE iteration {iteration}: vocab_size={current_vocab_size}")
    
    print(f"Final vocabulary size: {len(vocab)}")
    return vocab

def save_vocab(vocab, output_path):
    """Save vocabulary to JSON."""
    os.makedirs(output_path, exist_ok=True)
    vocab_path = os.path.join(output_path, 'vocab.json')
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary saved to: {vocab_path}")
    return vocab_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Sinhala vocabulary for XTTS")
    parser.add_argument("--metadata_path", required=True, help="Path to metadata_train.csv")
    parser.add_argument("--output_path", default="checkpoints/", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=2500, help="Target vocabulary size")
    
    args = parser.parse_args()
    
    vocab = extract_sinhala_vocab(args.metadata_path, args.vocab_size)
    save_vocab(vocab, args.output_path)
    
    print("\n✓ Sinhala vocabulary created successfully!")
    print(f"  Total tokens: {len(vocab)}")
