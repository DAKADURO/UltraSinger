"""Whisper Speech Recognition Module"""
import json
import os
import inspect
import textwrap

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

from modules.Audio.chunk_planner import plan_chunks
from modules.gpu_memory import free_gpu_memory, resolve_whisper_settings
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.lyrics import map_lyrics_to_segments
from modules.Speech_Recognition.syllable_timing import assign_chars_to_words
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper

#Addition for numbers to words
import re
import ast
from num2words import num2words

#Addition for numbers to words
re_split_preserve_space = re.compile(r'(\d+|\W+|\w+)')


SAMPLE_RATE = 16000  # whisperx.load_audio always resamples to 16 kHz
DEFAULT_CHUNK_SECONDS = 60.0
LYRICS_LOW_MATCH_RATIO = 0.3

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
    LARGE_V3_TURBO = "large-v3-turbo"

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

def _is_out_of_memory(exception: Exception) -> bool:
    """True for CUDA out of memory errors from torch (alignment) and ctranslate2 (whisper)"""
    return isinstance(exception, OutOfMemoryError) or "out of memory" in str(exception).lower()


class _WhisperRunner:
    """Whisper model that lowers its memory needs and retries when the GPU runs out of memory.

    Order of the fallbacks: halve the batch size, switch float16 to int8_float16, use the cpu.
    """

    def __init__(self, model: WhisperModel, device: str, compute_type: str, batch_size: int,
                 language: str | None):
        self.model_name = model.value
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self.model = None
        self._load()

    def _load(self) -> None:
        free_gpu_memory()
        print(
            f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(self.model_name)}, "
            f"{red_highlighted(self.device)} as worker, {self.compute_type} and batch size {self.batch_size}"
        )
        self.model = whisperx.load_model(
            self.model_name, language=self.language, device=self.device, compute_type=self.compute_type
        )

    def _release(self) -> None:
        self.model = None
        free_gpu_memory()

    def _lower_memory_needs(self) -> bool:
        """Apply the next fallback, False if there is nothing left to try"""
        if self.batch_size > 1:
            self.batch_size = max(1, self.batch_size // 2)
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Out of GPU memory')}, retrying with batch size {self.batch_size}")
            free_gpu_memory()
            return True
        if self.device == "cuda" and self.compute_type in ("float16", "float32"):
            self.compute_type = "int8_float16"
        elif self.device == "cuda":
            self.device, self.compute_type = "cpu", "int8"
        else:
            return False
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Out of GPU memory')}, reloading whisper with {self.device}/{self.compute_type}")
        self._release()
        self._load()
        return True

    def transcribe(self, audio):
        while True:
            try:
                return self.model.transcribe(audio, batch_size=self.batch_size, language=self.language)
            except Exception as exception:
                if not _is_out_of_memory(exception):
                    raise
                message = str(exception)
            # Handled outside of the except block, so the failed tensors are no longer referenced
            if not self._lower_memory_needs():
                raise OutOfMemoryError(message)

    def close(self) -> None:
        self._release()


def _checkpoint_path(checkpoint_dir: str | None, index: int, start: float, end: float) -> str | None:
    if checkpoint_dir is None:
        return None
    return os.path.join(checkpoint_dir, f"chunk_{index:03d}_{start:.2f}_{end:.2f}.json")


def _transcribe_chunks(runner: _WhisperRunner, audio, chunks: list[tuple[float, float]],
                       checkpoint_dir: str | None) -> tuple[list[dict], str | None]:
    """Run whisper chunk by chunk, saving every finished chunk so an interrupted run can resume"""
    segments = []
    detected_language = runner.language
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for index, (start, end) in enumerate(chunks):
        path = _checkpoint_path(checkpoint_dir, index, start, end)
        if path is not None and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                saved = json.load(file)
            print(f"{ULTRASINGER_HEAD} Chunk {index + 1}/{len(chunks)} reused from cache")
        else:
            print(f"{ULTRASINGER_HEAD} Transcribing chunk {index + 1}/{len(chunks)} ({start:.1f}s - {end:.1f}s)")
            chunk_audio = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            result = runner.transcribe(chunk_audio)
            saved = {
                "language": result["language"],
                "segments": [
                    {**segment, "start": segment["start"] + start, "end": segment["end"] + start}
                    for segment in result["segments"]
                ],
            }
            if path is not None:
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(saved, file)
            free_gpu_memory()

        if detected_language is None:
            detected_language = saved["language"]
            # Use the same language for all following chunks
            runner.language = detected_language
        segments.extend(saved["segments"])

    return segments, detected_language


def _align_segments(segments: list[dict], audio, language: str, alignment_model: str | None, device: str) -> dict:
    """Align the text to the audio. Falls back to the cpu if the GPU is out of memory"""
    devices = [device, "cpu"] if device == "cuda" else [device]
    for current_device in devices:
        model_a = None
        try:
            free_gpu_memory()
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language, device=current_device, model_name=alignment_model
                )
            except ValueError as ve:
                print(
                    f"{red_highlighted(f'{ve}')}"
                    f"\n"
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                    f"Try add it with --align_model [huggingface]."
                )
                raise ve
            # Character timings are used to time the syllables
            return whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                current_device,
                return_char_alignments=True,
            )
        except Exception as exception:
            if not _is_out_of_memory(exception) or current_device == "cpu":
                raise
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Out of GPU memory')} while aligning, retrying on the cpu")
        finally:
            model_a = None
            free_gpu_memory()


def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int | None = None,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
    lyrics_words: list[str] | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    checkpoint_dir: str | None = None,
) -> TranscriptionResult:
    """Transcribe with whisper

    The audio is cut into chunks inside silences and transcribed one by one, and the whisper
    model is unloaded before the alignment model is loaded, so both never share the GPU memory.
    With lyrics_words the recognised text is replaced by the known lyrics before the alignment.
    """
    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.
    batch_size, compute_type = resolve_whisper_settings(device, batch_size, compute_type)
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    runner = None
    try:
        audio = whisperx.load_audio(audio_path)
        chunks = plan_chunks(len(audio) / SAMPLE_RATE, _get_silences(audio_path), chunk_seconds)

        runner = _WhisperRunner(model, device, compute_type, batch_size, language)
        print(f"{ULTRASINGER_HEAD} Transcribing {audio_path} in {len(chunks)} chunk(s)")
        segments, detected_language = _transcribe_chunks(runner, audio, chunks, checkpoint_dir)
        runner.close()
        runner = None

        language = language or detected_language

        if lyrics_words:
            segments = _apply_lyrics(segments, lyrics_words)

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False:
            for obj in segments:
                obj["text"] = number_to_words(obj["text"], language)

        result_aligned = _align_segments(segments, audio, language, alignment_model, device)
        transcribed_data = convert_to_transcribed_data(result_aligned)

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
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
        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        if exception.args and "CUDA failed with error out of memory" in str(exception.args[0]):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception
    finally:
        # Restore original torch.load after the models are loaded, so other modules
        # (like pitch detection) are not affected by the monkey-patch
        torch.load = _original_torch_load
        if runner is not None:
            runner.close()
        free_gpu_memory()


def _get_silences(audio_path: str) -> list[tuple[float, float]]:
    # Imported here: the silence module pulls in pydub and librosa
    from modules.Audio.silence_processing import get_silence_sections
    return get_silence_sections(audio_path)


def _apply_lyrics(segments: list[dict], lyrics_words: list[str]) -> list[dict]:
    """Replace the recognised text of every segment with the matching part of the known lyrics"""
    mapped, match_ratio = map_lyrics_to_segments([seg["text"].split() for seg in segments], lyrics_words)
    print(f"{ULTRASINGER_HEAD} Using given lyrics, {round(match_ratio * 100)}% of the recognised words matched them")
    if match_ratio < LYRICS_LOW_MATCH_RATIO:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} the lyrics hardly match the recognised words, "
              f"check that they belong to this song and language")
    return [{**segment, "text": " ".join(words)} for segment, words in zip(segments, mapped) if words]


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        words = segment["words"]
        chars_per_word = None
        if segment.get("chars"):
            chars_per_word = assign_chars_to_words([obj["word"] for obj in words], segment["chars"])

        for position, obj in enumerate(words):
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if chars_per_word is not None:
                vtd.char_timings = _compact_char_timings(chars_per_word[position])
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
                vtd.char_timings = None  # the characters belong to a word that was not timed
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data


def _compact_char_timings(chars: list[dict]) -> list[dict]:
    """Keep only what the syllable timing needs, as plain floats so it can be saved as json"""
    return [
        {key: (float(value) if key in ("start", "end") else value)
         for key, value in char.items() if key in ("char", "start", "end")}
        for char in chars
    ]
