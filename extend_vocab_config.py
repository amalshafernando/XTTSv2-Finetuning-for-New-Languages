#!/usr/bin/env python3
"""Sinhala-specific vocabulary extension for XTTS"""

import json
import os
import argparse
import pandas as pd
from collections import Counter

def extract_sinhala_characters(text):
    """Extract only Sinhala Unicode characters (U+0D80-U+0DFF)"""
    sinhala_chars = []
    for char in text:
        code_point = ord(char)
        if 0x0D80 <= code_point <= 0x0DFF:
            sinhala_chars.append(char)
    return sinhala_chars

def create_sinhala_vocab(metadata_path, output_path, vocab_size=2500):
    """Create BPE vocabulary specifically for Sinhala"""
    
    # Read metadata
    df = pd.read_csv(metadata_path, sep="|", header=None,
                     names=["audio_file", "text", "speaker"])
    texts = df.text.tolist()
    
    print(f"Processing {len(texts)} Sinhala texts...")
    
    # Extract character-level vocabulary
    char_freq = Counter()
    bigram_freq = Counter()
    word_freq = Counter()
    
    for text in texts:
        sinhala_text = ''.join(extract_sinhala_characters(text))
        
        for char in sinhala_text:
            char_freq[char] += 1
        
        for i in range(len(sinhala_text) - 1):
            bigram = sinhala_text[i:i+2]
            bigram_freq[bigram] += 1
        
        for word in text.split():
            word_freq[word] += 1
    
    # Build vocabulary
    vocab = {}
    idx = 0
    
    # Special tokens
    special_tokens = ["[PAD]", "[UNK]", "[START]", "[STOP]", "[si]"]
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # Add all Sinhala characters
    print(f"Found {len(char_freq)} unique Sinhala characters")
    for char in sorted(char_freq.keys(), key=lambda x: char_freq[x], reverse=True):
        vocab[char] = idx
        idx += 1
        if idx >= vocab_size * 0.7:
            break
    
    # Add high-frequency bigrams (✅ FIX: Use x[1] for count)
    print(f"Found {len(bigram_freq)} unique bigrams")
    for bigram, count in sorted(bigram_freq.items(),
                                key=lambda x: x[1], reverse=True):  # ✅ FIXED
        if idx >= vocab_size:
            break
        if bigram not in vocab and count >= 2:
            vocab[bigram] = idx
            idx += 1
    
    # Add common words
    for word, count in sorted(word_freq.items(),
                              key=lambda x: x[1], reverse=True):  # ✅ FIXED
        if idx >= vocab_size:
            break
        if word not in vocab and count >= 1:
            vocab[word] = idx
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
    print(f"   - Sinhala characters: {len(char_freq)}")
    print(f"   - Character bigrams: {sum(1 for b in bigram_freq if bigram_freq[b] >= 2)}")
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
        
        if "si" not in config["languages"]:
            config["languages"].append("si")
        
        if "model_args" in config:
            config["model_args"]["gpt_num_text_tokens"] = args.vocab_size
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Updated config.json with language 'si'")
    except Exception as e:
        print(f"⚠️ Could not update config.json: {e}")

if __name__ == "__main__":
    # ✅ FIXED: Correct argument parser with right arguments
    parser = argparse.ArgumentParser(description="Create Sinhala vocabulary for XTTS")
    
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to metadata_train.csv")
    parser.add_argument("--output_path", type=str, default="checkpoints/",
                       help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=2500,
                       help="Target vocabulary size")
    
    args = parser.parse_args()
    
    # ✅ FIXED: Call the correct function
    vocab_path = create_sinhala_vocab(
        args.metadata_path,
        args.output_path,
        args.vocab_size
    )
    
    adjust_config(args)
    
    print(f"\n✅ Sinhala tokenization complete!")
    print(f"   Use vocab.json in your XTTS training config")
