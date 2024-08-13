#!/usr/bin/env python3

# Copyright (c) 2024 Julian Müller (ChaoticByte)

# Disable FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports

from argparse import ArgumentParser
from pathlib import Path
from subprocess import check_call, DEVNULL
from tempfile import TemporaryDirectory
from typing import List

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from transformers import pipeline

# Some constant variables

NLP_MODEL = "facebook/bart-large-cnn"
root_dir = Path(__file__).parent
whisper_cpp_binary = (root_dir / "vendor" / "whisper.cpp" / "main").__str__()

# Steps

def convert_audio(media_file: str, output_file: str):
    '''Convert media to mono 16kHz pcm_s16le wav using ffmpeg'''
    check_call([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", media_file,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_file])

def transcribe(model_file: str, audio_file: str, output_file: str):
    '''Transcribe audio file using whisper.cpp'''
    check_call([
        whisper_cpp_binary,
        "-m", model_file,
        "--max-context", "64",
        "--beam-size", "5",
        "--no-prints",
        "--no-timestamps",
        "--output-txt",
        "--output-file", output_file[:-4], # strip '.txt' file ending
        audio_file], stdout=DEVNULL) 

def cleanup_text(t: str) -> str:
    t = t.replace("\n", "")
    t = t.replace("\r", "")
    t = t.strip()
    return t

def split_text(t: str, max_tokens: int) -> List[str]:
    '''Split text into semantic segments'''
    tokenizer = Tokenizer.from_pretrained(NLP_MODEL)
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer, (int(max_tokens*0.8), max_tokens))
    chunks = splitter.chunks(t)
    return chunks

def summarize(chunks: List[str], summary_min: int, summary_max: int) -> str:
    '''Summarize all segments (chunks) using a language model'''
    chunks_summarized = []
    summ = pipeline("summarization", model=NLP_MODEL)
    for c in chunks:
        chunks_summarized.append(
            summ(c, max_length=summary_max, min_length=summary_min, do_sample=False)[0]['summary_text'].strip())
    return "\n".join(chunks_summarized)

# Main

if __name__ == "__main__":
    # parse commandline arguments
    argp = ArgumentParser()
    argp.add_argument("--summin", metavar="n", type=int, default=10, help="The minimum lenght of a segment summary [10, min: 5]")
    argp.add_argument("--summax", metavar="n", type=int, default=90, help="The maximum lenght of a segment summary [90, min: 5]")
    argp.add_argument("--segmax", metavar="n", type=int, default=375, help="The maximum number of tokens per segment [375, 5 - 500]")
    argp.add_argument("-m", required=True, metavar="filepath", type=Path, help="The path to a whisper.cpp-compatible model file")
    argp.add_argument("-i", required=True, metavar="filepath", type=Path, help="The path to the media file")
    argp.add_argument("-o", required=True, metavar="filepath", type=Path, help="Where to save the output text to")
    args = argp.parse_args()
    # Clamp values
    args.summin = max(5, args.summin)
    args.summax = max(5, args.summax)
    args.segmax = max(5, min(args.segmax, 500))
    # create tmpdir
    with TemporaryDirectory(suffix="as") as d:
        converted_audio_path = (Path(d) / "audio.wav").__str__()
        transcript_path = (Path(d) / "transcript.txt").__str__()
        # convert using ffmpeg
        print("* Converting media to the correct format ...")
        convert_audio(args.i.__str__(), converted_audio_path)
        # transcribe
        print("* Transcribing audio ...")
        transcribe(args.m.__str__(), converted_audio_path, transcript_path)
        # read transcript
        text = Path(transcript_path).read_text()
    # cleanup text & summarize
    print("* Summarizing transcript ...")
    text = cleanup_text(text)
    chunks = split_text(text, args.segmax)
    summary = summarize(chunks, args.summin, args.summax)
    print(f"\n{summary}\n")
    print(f"* Saving summary to {args.o.__str__()}")
    with args.o.open("w+") as f: # overwrites
        f.write(summary)
