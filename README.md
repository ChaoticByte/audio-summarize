# audio-summarize

An audio summarizer that glues together ffmpeg, whisper.cpp and BART.

## Dependencies

- Python 3 (tested: 3.12)
- ffmpeg
- git
- make & c/c++ compiler

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
audio-summarize.py -m filepath -i filepath -o filepath
                    [--summin n] [--summax n] [--segmax n]

options:
  -h, --help   show this help message and exit
  --summin n   The minimum lenght of a segment summary [10]
  --summax n   The maximum lenght of a segment summary [90]
  --segmax n   The maximum number of tokens per segment [375, max: 500]
  -m filepath  The path to a whisper.cpp-compatible model file
  -i filepath  The path to the media file
  -o filepath  Where to save the output text to
```

Example:

```bash
./audio-summarize.py -m ./tmp/whisper_ggml-small.en-q5_1.bin -i ./tmp/test.webm -o ./tmp/output.txt
```
