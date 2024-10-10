# Live Transcription Project

This project provides a Python-based solution for real-time audio transcription using multiple libraries for speech processing, AI, and machine learning. It supports both Linux and Windows environments and includes GPU acceleration with CUDA 12.2 for enhanced performance.

## Features
- Real-time transcription using OpenAI's Whisper and Faster Whisper.
- Voice activity detection with Silero VAD.
- Pre-trained models for NLP and speech recognition with Huggingface Transformers.
- Speech processing and feature extraction using Librosa and PyTorch.
- Support for both CPU and GPU (CUDA 12.2) environments.

## Prerequisites

### CUDA (Optional for GPU support)
To enable GPU acceleration, you need to install the NVIDIA CUDA Toolkit. This project is compatible with:
- **CUDA 12.2** (for GPU acceleration)
- Ensure you have CUDA drivers installed for your GPU. Refer to [CUDA installation instructions](https://developer.nvidia.com/cuda-toolkit) for details.

### Python Version
This project requires **Python 3.8 or higher**.

## Installation

### Step 1: Clone the repository
First, clone the repository from GitHub:
    ```bash
    git clone https://github.com/yourusername/yourprojectname.git
    cd yourprojectname
    ```

### Step 2: Install the required libraries

The libraries used in this project are listed in the requirements.txt file. You can install them using pip.
#### For Linux (with CUDA 12.2 support)
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
```
#### For Windows (with CUDA 12.2 support)
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
```
If you don't need GPU support (CPU-only), you can install PyTorch without specifying the CUDA version:
```bash
pip install torch torchvision torchaudio
```

### Step 3: Verify Installation
You can verify that PyTorch is using your GPU (if available) with this simple Python command:
```python

import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
```

### Step 4: Run the Project
Once all dependencies are installed, you can run the live transcription script:
```bash
python live_transcription.py
```
Ensure that your audio input device (e.g., microphone) is properly configured for real-time processing.

## Supported Libraries
- [PyTorch](https://pytorch.org/) - For machine learning and deep learning models.
- [Librosa](https://librosa.org/) - For audio feature extraction.
- [SpeechBrain](https://speechbrain.github.io/) - For speech processing.
- [Whisper](https://github.com/openai/whisper) - For transcription using OpenAI models.
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized transcription models.
- [Silero VAD](https://github.com/snakers4/silero-vad) - For voice activity detection.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2): Optimized inference engine supporting CUDA.
- [Transformers]: For pre-trained NLP models.
- [FastText](https://fasttext.cc/): For fast and efficient text classification and word representation.
- [SentencePiece](https://github.com/google/sentencepiece): For tokenization.
- [NLTK](https://www.nltk.org/): For natural language processing.
- [OpenAI Whisper](https://github.com/openai/whisper): For transcription using Whisper models.
- And more...

## Contribution

Feel free to contribute to this project by opening a pull request or reporting issues.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

```javascript
This `README.md` should now display correctly, with all sections and code blocks formatted as intended. It includes the installation instructions for both Linux and Windows with CUDA 12.2 support, ensuring the document flows smoothly without disjointed parts.
```


