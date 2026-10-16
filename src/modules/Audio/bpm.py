import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def fold_tempo(bpm: float, low: float = 75.0, high: float = 150.0) -> float:
    """Move a tempo by whole octaves (x2 / :2) into [low, high).

    Tempo estimators often return half or double of the felt tempo. The syllable splitting uses
    sixteenth notes of this tempo, so a wrong octave makes it four times too coarse or too fine.
    """
    if bpm <= 0:
        return bpm
    while bpm >= high:
        bpm /= 2
    while bpm < low:
        bpm *= 2
    return bpm


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data"""
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sampling_rate)
    bpm = float(fold_tempo(wav_tempo[0]))

    print(f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(bpm, 2)))}")
    return bpm


def get_bpm_from_file(wav_file: str) -> float:
    """Get real bpm from audio file"""
    data, sampling_rate = sf.read(wav_file, dtype='float32')
    # Transpose if stereo to match librosa's expected format
    if len(data.shape) > 1:
        data = data.T
    # Convert to mono if stereo
    if data.ndim > 1:
        data = librosa.to_mono(data)
    return get_bpm_from_data(data, sampling_rate)
