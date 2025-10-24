# XTTSv2 Sinhala Language Finetuning

This repository contains the code and setup for fine-tuning XTTSv2 (eXtended Text-to-Speech) for Sinhala language.

## Dataset

The Sinhala TTS dataset is available on Kaggle: [Sinhala TTS Dataset](https://www.kaggle.com/datasets/amalshaf/sinhala-tts-dataset)

## Quick Start on Kaggle

### 1. Upload Dataset to Kaggle
- Upload your Sinhala dataset to Kaggle as a dataset
- Make sure the dataset contains:
  - `wavs/` folder with audio files
  - `train.csv` with training metadata
  - `val.csv` with validation metadata
  - `dict.txt` with vocabulary (optional)

### 2. Create Kaggle Notebook
- Create a new notebook on Kaggle
- Enable GPU (P100 or better recommended)
- Upload this repository to Kaggle or clone from GitHub

### 3. Run Training
```python
# Clone this repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Change to repository directory
import os
os.chdir('YOUR_REPO_NAME')

# Run training
!python kaggle_training.py
```

## Files Structure

```
├── kaggle_dataset_setup.py      # Dataset preparation script
├── kaggle_training.py           # Main training pipeline
├── kaggle_notebook_template.ipynb  # Kaggle notebook template
├── XTTSv2-Finetuning-for-New-Languages/
│   ├── train_gpt_xtts.py        # GPT training script
│   ├── train_dvae_xtts.py       # DVAE training script
│   ├── extend_vocab_config.py   # Vocabulary extension
│   └── requirements_kaggle.txt  # Kaggle-compatible requirements
└── README.md                    # This file
```

## Training Process

The training consists of two main steps:

### 1. DVAE Training (Optional but Recommended)
- Fine-tunes the Discrete VAE component
- Helps with audio quality for new languages
- Takes about 1-2 hours on Kaggle GPU

### 2. GPT Training (Main Model)
- Fine-tunes the main GPT model for Sinhala
- This is the core training process
- Takes about 4-8 hours on Kaggle GPU

## Usage After Training

Once training is complete, you can use the model:

```python
from TTS.api import TTS

# Load your finetuned model
tts = TTS("checkpoints/run/training")

# Generate Sinhala speech
text = "ආයුබෝවන්"  # "Hello" in Sinhala
speaker_wav = "path/to/reference/audio.wav"

# Generate speech
tts.tts_to_file(
    text=text, 
    speaker_wav=speaker_wav, 
    language="si", 
    file_path="output.wav"
)
```

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- At least 20GB disk space

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size in training scripts
2. **Dataset Not Found**: Check Kaggle dataset path and permissions
3. **Model Download Failed**: Check internet connection in Kaggle
4. **Training Stuck**: Check GPU availability and memory usage

### Performance Tips:

1. Use P100 or better GPU on Kaggle
2. Enable mixed precision training
3. Use gradient accumulation for larger effective batch sizes
4. Monitor GPU memory usage during training

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project follows the same license as the original XTTSv2 project.
