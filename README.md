# audio-summarize

An audio summarizer that glues together [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [BART](https://huggingface.co/facebook/bart-large-cnn).

## Dependencies

- Python 3 (tested: 3.12)

## Setup

Create a virtual environment for python, activate it and install the required python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Run

1. In your terminal, make shure you have your python venv activated
2. Run audio-summarize.py

### Usage

```
./audio-summarize.py -i filepath -o filepath
                     [--summin n] [--summax n] [--segmax n]
                     [--lang lang] [-m name]

options:
  -h, --help   show this help message and exit
  --summin n   The minimum lenght of a segment summary [10, min: 5]
  --summax n   The maximum lenght of a segment summary [90, min: 5]
  --segmax n   The maximum number of tokens per segment [375, 5 - 500]
  --lang lang  The language of the audio source ['en']
  -m name      The name of the whisper model to be used ['small.en']
  -i filepath  The path to the media file
  -o filepath  Where to save the output text to
```

Example:

```bash
./audio-summarize.py -i ./tmp/test.webm -o ./tmp/output.txt
```

## How does it work?

To summarize a media file, the program executes the following steps:

1. Convert and transcribe the media file using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), using [ffmpeg](https://www.ffmpeg.org/) and [ctranslate2](https://github.com/OpenNMT/CTranslate2/) under the hood
2. Semantically split up the transcript into segments using [semantic-text-splitter](https://github.com/benbrandt/text-splitter) and the tokenizer for BART
3. Summarize each segment using BART ([`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn))
4. Write the results to a text file
