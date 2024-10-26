#!/bin/bash
pip install --timeout=120 -r requirements.txt
pip install --timeout=120 --upgrade faster-whisper

python -c "import nltk; nltk.download('punkt_tab')"