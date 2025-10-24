# XTTSv2 Sinhala Finetuning

This repository contains the finetuning setup for XTTSv2 model for Sinhala language.

## Dataset
- Training samples: 1002
- Validation samples: 253
- Audio files: 1251 WAV files
- Language: Sinhala

## Files Structure
datasets/
└── sinhala/
├── wavs/ (audio files)
├── metadata_train.csv
└── metadata_eval.csv



## Quick Start on Kaggle

1. Upload this repository to Kaggle
2. Open `kaggle_training_notebook.ipynb`
3. Run all cells sequentially
4. The model will be saved in `checkpoints/` directory

## Files Description

- `setup_kaggle.py`: Setup script for Kaggle environment
- `requirements_kaggle.txt`: Kaggle-compatible requirements
- `kaggle_training_notebook.ipynb`: Complete training notebook
- `extend_vocab_config.py`: Vocabulary extension for Sinhala
- `train_dvae_xtts.py`: DVAE finetuning script
- `train_gpt_xtts.py`: Main GPT finetuning script

## Training Parameters

- Language: Sinhala (si)
- Extended vocabulary size: 2000
- Batch size: 4 (Kaggle optimized)
- Learning rate: 5e-6
- Max text length: 300 characters
- Max audio length: 250,000 samples
