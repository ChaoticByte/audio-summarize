#!/usr/bin/env python3

# Copyright (c) 2024 Julian Müller (ChaoticByte)

# Disable FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports

from argparse import ArgumentParser
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from transformers import pipeline


# Transcription

def transcribe(model_name: str, audio_file: str, language: str) -> str:
    '''Transcribe the media using faster-whisper'''
    t_chunks = []
    print("* Loading model ", end="", flush=True)
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio_file, language=language, beam_size=5, condition_on_previous_text=False)
    print()
    print("* Transcribing audio ", end="", flush=True)
    for s in segments:
        print(".", end="", flush=True)
        t_chunks.append(s.text)
    print()
    t = "".join(t_chunks)
    return t


# NLP

NLP_MODEL = "facebook/bart-large-cnn"

def split_text(t: str, max_tokens: int) -> List[str]:
    '''Split text into semantic segments'''
    print("* Splitting up transcript into semantic segments")
    tokenizer = Tokenizer.from_pretrained(NLP_MODEL)
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer, (int(max_tokens*0.8), max_tokens))
    chunks = splitter.chunks(t)
    return chunks

def summarize(chunks: List[str], summary_min: int, summary_max: int) -> str:
    '''Summarize all segments (chunks) using a language model'''
    print("* Summarizing transcript segments ", end="", flush=True)
    chunks_summarized = []
    summ = pipeline("summarization", model=NLP_MODEL)
    for c in chunks:
        print(".", end="", flush=True)
        chunks_summarized.append(
            summ(c, max_length=summary_max, min_length=summary_min, do_sample=False)[0]['summary_text'].strip())
    print()
    return "\n".join(chunks_summarized)


# Main

if __name__ == "__main__":
    # parse commandline arguments
    argp = ArgumentParser()
    argp.add_argument("--summin", metavar="n", type=int, default=10, help="The minimum lenght of a segment summary [10, min: 5]")
    argp.add_argument("--summax", metavar="n", type=int, default=90, help="The maximum lenght of a segment summary [90, min: 5]")
    argp.add_argument("--segmax", metavar="n", type=int, default=375, help="The maximum number of tokens per segment [375, 5 - 500]")
    argp.add_argument("--lang", metavar="lang", type=str, default="en", help="The language of the audio source ['en']")
    argp.add_argument("-m", metavar="name", type=str, default="small.en", help="The name of the whisper model to be used ['small.en']")
    argp.add_argument("-i", required=True, metavar="filepath", type=Path, help="The path to the media file")
    argp.add_argument("-o", required=True, metavar="filepath", type=Path, help="Where to save the output text to")
    args = argp.parse_args()
    # Clamp values
    args.summin = max(5, args.summin)
    args.summax = max(5, args.summax)
    args.segmax = max(5, min(args.segmax, 500))
    # transcribe
    text = transcribe(args.m, args.i, args.lang).strip()
    # split up into semantic segments & summarize
    chunks = split_text(text, args.segmax)
    summary = summarize(chunks, args.summin, args.summax)
    print(f"\n{summary}\n")
    print(f"* Saving summary to {args.o.__str__()}")
    with args.o.open("w+") as f: # overwrites
        f.write(summary)
