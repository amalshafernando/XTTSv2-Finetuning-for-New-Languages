#!/usr/bin/env python3
"""
Sinhala-specific vocabulary extension for XTTS
Handles Sinhala abugida script (U+0D80-U+0DFF)
"""

import json
import os
import argparse
import pandas as pd
from collections import defaultdict, Counter

def extract_sinhala_characters(text):
    """Extract only Sinhala Unicode characters"""
    # Sinhala Unicode range: U+0D80 to U+0DFF
    sinhala_chars = []
    for char in text:
        code_point = ord(char)
        if 0x0D80 <= code_point <= 0x0DFF:  # Sinhala range
            sinhala_chars.append(char)
    return sinhala_chars

def create_sinhala_vocab(metadata_path, output_path, vocab_size=2500):
    """
    Create BPE vocabulary specifically for Sinhala
    """
    
    # Read metadata
    df = pd.read_csv(metadata_path, sep="|", header=None, 
                     names=["audio_file", "text", "speaker"])
    texts = df.text.tolist()
    
    print(f"Processing {len(texts)} Sinhala texts...")
    
    # Step 1: Extract character-level vocabulary
    char_freq = Counter()
    word_freq = Counter()
    bigram_freq = Counter()
    
    for text in texts:
        # Extract Sinhala characters
        sinhala_text = ''.join(extract_sinhala_characters(text))
        
        # Count individual characters
        for char in sinhala_text:
            char_freq[char] += 1
        
        # Count character bigrams (adjacent pairs)
        for i in range(len(sinhala_text) - 1):
            bigram = sinhala_text[i:i+2]
            bigram_freq[bigram] += 1
        
        # Count space-separated words (for reference)
        for word in text.split():
            word_freq[word] += 1
    
    # Step 2: Build vocabulary with character + high-freq bigrams
    vocab = {}
    idx = 0
    
    # Add special tokens
    special_tokens = ["[PAD]", "[UNK]", "[START]", "[STOP]", "[si]"]
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # Add all individual Sinhala characters
    print(f"Found {len(char_freq)} unique Sinhala characters")
    for char in sorted(char_freq.keys(), key=lambda x: char_freq[x], reverse=True):
        vocab[char] = idx
        idx += 1
        if idx >= vocab_size * 0.7:  # Reserve 30% for bigrams
            break
    
    # Add high-frequency bigrams
    print(f"Found {len(bigram_freq)} unique bigrams")
    for bigram, count in sorted(bigram_freq.items(), 
                                 key=lambda x: x, reverse=True):
        if idx >= vocab_size:
            break
        if bigram not in vocab and count >= 2:  # Min frequency threshold
            vocab[bigram] = idx
            idx += 1
    
    # Add common space-separated words (fallback)
    for word, count in sorted(word_freq.items(), 
                              key=lambda x: x, reverse=True):
        if idx >= vocab_size:
            break
        if word not in vocab and count >= 1:
            vocab[word] = idx
            idx += 1
    
    print(f"✅ Final vocabulary size: {len(vocab)} tokens")
    
    # Step 3: Save vocabulary
    vocab_dir = os.path.join(output_path, "XTTS-v2")
    os.makedirs(vocab_dir, exist_ok=True)
    
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved vocabulary to: {vocab_path}")
    
    # Step 4: Print statistics
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   - Special tokens: {len(special_tokens)}")
    print(f"   - Sinhala characters: {len(char_freq)}")
    print(f"   - Character bigrams: {sum(1 for b in bigram_freq if bigram_freq[b] >= 2)}")
    print(f"   - Total: {len(vocab)} tokens")
    
    return vocab_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Sinhala vocabulary for XTTS")
    parser.add_argument("--metadata_path", type=str, required=True, 
                       help="Path to metadata_train.csv")
    parser.add_argument("--output_path", type=str, default="checkpoints/", 
                       help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=2500, 
                       help="Target vocabulary size")
    
    args = parser.parse_args()
    
    vocab_path = create_sinhala_vocab(
        args.metadata_path, 
        args.output_path, 
        args.vocab_size
    )
    
    print(f"\n✅ Sinhala tokenization complete!")
    print(f"   Use this vocab.json in your XTTS config")
