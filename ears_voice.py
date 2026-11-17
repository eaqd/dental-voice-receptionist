"""Ears and voice, both free and on the CPU: faster-whisper turns speech into text, Kokoro turns text into speech."""
import os
import wave
from functools import lru_cache

import numpy as np

SR = 24000  # Kokoro's sample rate
KOKORO_DIR = os.getenv("KOKORO_DIR", os.path.join(os.path.expanduser("~"), ".cache", "kokoro"))


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel
    # base.en: about 150 MB, downloaded on first use; small.en hears better and is slower
    return WhisperModel(os.getenv("WHISPER_MODEL", "base.en"), device="cpu", compute_type="int8")


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Whisper listens at 16 kHz. A light smoothing first keeps high sounds from folding into speech."""
    if rate == 16000:
        return samples.astype(np.float32)
    if rate > 16000:
        samples = np.convolve(samples, np.ones(3, dtype=np.float32) / 3, mode="same")
    n = int(len(samples) * 16000 / rate)
    return np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)


def hear(wav_path) -> str:
    # We read the WAV ourselves and hand Whisper the samples. Passing a file path makes faster-whisper 1.2.1 decode
    # it with PyAV, and PyAV 19 (what pip installs today) rejects one of its options (found building this, 2026-11-16).
    samples, rate = load_wav(wav_path)
    if len(samples) < rate // 10:  # under a tenth of a second: nothing to hear
        return ""
    segments, _ = _whisper().transcribe(to_16k(samples, rate), language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    return Kokoro(os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx"), os.path.join(KOKORO_DIR, "voices-v1.0.bin"))


def speak(text: str, voice: str = "bf_emma", speed: float = 1.0) -> np.ndarray:
    """British voices start with b (bf_emma), American with a (af_heart)."""
    samples, _ = _kokoro().create(text, voice=voice, speed=speed, lang="en-gb" if voice.startswith("b") else "en-us")
    return np.asarray(samples, dtype=np.float32)


def save_wav(path, samples: np.ndarray, rate: int = SR):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def load_wav(path) -> tuple[np.ndarray, int]:
    """Any plain PCM WAV (8, 16, 24 or 32-bit, mono or stereo) as float32 mono in -1..1, and its sample rate."""
    with wave.open(str(path), "rb") as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:  # 8-bit WAV is unsigned
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        data = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif width == 3:  # 24-bit: pad each sample to 32 bits
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        data = (np.pad(b, ((0, 0), (1, 0))).view("<i4").ravel().astype(np.float32)) / 2 ** 31
    elif width == 4:
        data = np.frombuffer(raw, np.int32).astype(np.float32) / 2 ** 31
    else:
        raise ValueError(f"{path}: {width * 8}-bit WAV is not supported")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate
