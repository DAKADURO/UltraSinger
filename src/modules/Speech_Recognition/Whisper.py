"""Whisper Speech Recognition Module"""
import copy
import gc
import os
import inspect
import textwrap
from typing import Optional

# Set environment variable to handle cuDNN loading issues
# CUDA_MODULE_LOADING=LAZY allows PyTorch to continue even if some cuDNN modules are not found
#os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch

# Fix for PyTorch 2.6+ compatibility with omegaconf and old models
# MUST be done before importing whisperx or any other modules that use torch.load
# Monkey-patch torch.load to use weights_only=False for compatibility with older models
# This is necessary because PyTorch 2.6+ changed the default to weights_only=True
# but WhisperX/Pyannote models were saved with pickle and require weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False regardless of what was passed
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.lyrics_client import (
    MIN_SCORE_GAIN,
    LyricLine,
    alignment_score,
    candidate_offsets,
    estimate_offset,
    lines_to_segments,
    trim_segment_ends,
)
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper

#Addition for numbers to words
import re
import ast
from num2words import num2words

#Addition for numbers to words
re_split_preserve_space = re.compile(r'(\d+|\W+|\w+)')


MEMORY_ERROR_MESSAGE = f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"

class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line,language='en'):
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    out_tokens = []
    in_tokens = re_split_preserve_space.findall(line)
    for token in in_tokens:
        try:
            num = ast.literal_eval(token)
            try:
                out_tokens.append(num2words(num, lang=language))
            except NotImplementedError:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language for number transcription. Keeping number as numeric characters for line: {line}, token: {token}"
                )
        except Exception:
            out_tokens.append(token)
    return ''.join(out_tokens) 

def replace_code_lines(source, start_token, end_token,
                       replacement, escape_tokens=True):
    """Replace the source code between `start_token` and `end_token`
    in `source` with `replacement`. The `start_token` portion is included
    in the replaced code. If `escape_tokens` is True (default),
    escape the tokens to avoid them being treated as a regular expression."""

    if escape_tokens:
        start_token = re.escape(start_token)
        end_token = re.escape(end_token)

    def replace_with_indent(match):
        indent = match.group(1)
        return textwrap.indent(replacement, indent)

    return re.sub(r"^(\s+)({}[\s\S]+?)(?=^\1{})".format(start_token, end_token),
                  replace_with_indent, source, flags=re.MULTILINE)

def __is_out_of_memory(error: Exception) -> bool:
    return isinstance(error, OutOfMemoryError) or "out of memory" in str(error).lower()


def __free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
    lyrics: Optional[list[LyricLine]] = None,
) -> TranscriptionResult:
    """Transcribe with whisper"""
    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.
    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    # Fixme: Why it is not none for cpu?
    #compute_type = "int8"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        while True:
            try:
                __free_gpu_memory()
                loaded_whisper_model = whisperx.load_model(
                    model.value, language=language, device=device, compute_type=compute_type
                )

                audio = whisperx.load_audio(audio_path)

                print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

                result = loaded_whisper_model.transcribe(
                    audio, batch_size=batch_size, language=language
                )
                break
            except RuntimeError as runtime_error:
                if device != "cuda" or not __is_out_of_memory(runtime_error):
                    raise
                loaded_whisper_model = None
                if batch_size > 4:
                    batch_size //= 2
                elif compute_type == "float16":
                    compute_type = "int8"
                elif batch_size > 1:
                    batch_size //= 2
                else:
                    raise
                print(
                    f"{ULTRASINGER_HEAD} Out of GPU memory, retrying with batch size {batch_size} and compute type {compute_type}"
                )

        # The transcription model is not needed anymore; free its memory before alignment
        loaded_whisper_model = None
        __free_gpu_memory()

        detected_language = result["language"]
        if language is None:
            language = detected_language

        # load alignment model and metadata
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=alignment_model
            )
        except ValueError as ve:
            print(
                f"{red_highlighted(f'{ve}')}"
                f"\n"
                f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                f"Try add it with --align_model [huggingface]."
            )
            raise ve

        def align_segments(segments):
            segments = copy.deepcopy(segments)
            #Addition for numbers to words (Using previous code from louispan in PR#135)
            if keep_numbers == False:
                for obj in segments:
                    obj["text"] = number_to_words(obj["text"], language)
            return whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )

        if lyrics:
            # Align the given lyrics instead of the transcribed text. The lyrics may be shifted against this
            # audio, so the shift estimated from the transcription is tried next to no shift at all and the
            # one that fits the audio better is used.
            audio_duration = len(audio) / 16000  # whisperx loads audio with 16 kHz
            print(f"{ULTRASINGER_HEAD} Aligning {blue_highlighted(str(len(lyrics)))} lyric lines")
            best_score, best_offset, result_aligned = None, 0.0, None
            for offset in candidate_offsets(estimate_offset(result["segments"], lyrics)):
                aligned = align_segments(lines_to_segments(lyrics, offset, audio_duration))
                score = alignment_score(aligned)
                print(f"{ULTRASINGER_HEAD} Time shift {offset:+.1f}s: alignment score {score:.3f}")
                if best_score is None or score > best_score + MIN_SCORE_GAIN:
                    best_score, best_offset, result_aligned = score, offset, aligned
            print(f"{ULTRASINGER_HEAD} Using time shift {blue_highlighted(f'{best_offset:+.1f}s')}")
            trim_segment_ends(result_aligned["segments"], audio)
        else:
            # align whisper output
            result_aligned = align_segments(result["segments"])

        transcribed_data = convert_to_transcribed_data(result_aligned)

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if "CUDA failed with error out of memory" in str(exception.args[0]):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception
    finally:
        # Restore original torch.load after models are loaded
        # This ensures other modules (like pitch detection) are not affected by the monkey-patch
        torch.load = _original_torch_load


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                #Addition for numbers to words (Using previous code from louispan in PR#135)
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1] if len(transcribed_data) != 0 else TranscribedData()
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                          f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
