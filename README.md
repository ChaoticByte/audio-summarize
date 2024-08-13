# audio-summarize

An audio summarizer that glues together ffmpeg, whisper.cpp and BART.

## Dependencies

- Python 3 (tested: 3.12)
- ffmpeg
- git
- make
- c/c++ compiler (on Ubuntu, installing `build-essential` does the trick)

## Setup

Create a virtual environment for python and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Run setup.sh

```bash
./setup.sh
```

## Run

1. You need a whisper.cpp compatible model file (-> https://huggingface.co/ggerganov/whisper.cpp)
2. In your terminal, make shure you have your python venv activated
3. Run audio-summarize.py

### Usage

```
./audio-summarize.py -m filepath -i filepath -o filepath
                   [--summin n] [--summax n] [--segmax n]

options:
  -h, --help   show this help message and exit
  --summin n   The minimum lenght of a segment summary [10, min: 5]
  --summax n   The maximum lenght of a segment summary [90, min: 5]
  --segmax n   The maximum number of tokens per segment [375, 5 - 500]
  -m filepath  The path to a whisper.cpp-compatible model file
  -i filepath  The path to the media file
  -o filepath  Where to save the output text to
```

Example:

```bash
./audio-summarize.py -m ./tmp/whisper_ggml-small.en-q5_1.bin -i ./tmp/test.webm -o ./tmp/output.txt
```

## How does it work?

To summarize a media file, the program executes the following steps:

1. Convert the media file with [ffmpeg](https://www.ffmpeg.org/) to a mono 16kHz 16bit-PCM wav file
2. Transcribe that wav file using [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
3. Clean up the transcript (newlines, whitespaces at the beginning and end)
4. Semantically split up the transcript into segments using [semantic-text-splitter](https://github.com/benbrandt/text-splitter) and the tokenizer for BART
5. Summarize each segment using BART ([`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn))
6. Write the results to a text file
