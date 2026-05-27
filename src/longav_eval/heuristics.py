from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def grayscale(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def image_quality_score(path: str | Path) -> tuple[float, dict[str, float]]:
    image = load_image(path)
    gray = grayscale(image)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float(np.var(gx) + np.var(gy))
    contrast = float(np.std(gray))
    exposure = float(1.0 - min(1.0, abs(float(np.mean(gray)) - 0.5) / 0.5))
    sharpness_norm = min(sharpness * 12.0, 1.0)
    contrast_norm = min(contrast * 4.0, 1.0)
    score = 100.0 * (0.45 * sharpness_norm + 0.30 * contrast_norm + 0.25 * exposure)
    return score, {
        "sharpness": sharpness_norm,
        "contrast": contrast_norm,
        "exposure": exposure,
    }


def color_histogram(path: str | Path, bins: int = 8) -> np.ndarray:
    image = load_image(path)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0.0, 1.0), density=False)
        histograms.append(hist.astype(np.float32))
    vector = np.concatenate(histograms)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity_score(path_a: str | Path, path_b: str | Path) -> float:
    vector_a = color_histogram(path_a)
    vector_b = color_histogram(path_b)
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0:
        return 0.0
    score = float(np.dot(vector_a, vector_b) / denom)
    return 100.0 * max(0.0, min(1.0, score))


def frame_motion_score(frame_paths: tuple[str, ...]) -> float:
    if len(frame_paths) < 2:
        return 0.0
    images = [grayscale(load_image(path)) for path in frame_paths]
    diffs = []
    for first, second in zip(images, images[1:]):
        diffs.append(float(np.mean(np.abs(first - second))))
    return 100.0 * min(1.0, float(np.mean(diffs)) * 8.0)


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def audio_quality_score(path: str | Path) -> tuple[float, dict[str, float]]:
    audio, _ = load_wav_mono(path)
    if audio.size == 0:
        return 0.0, {"rms": 0.0, "dynamic_range": 0.0, "clipping_penalty": 1.0}
    rms = float(np.sqrt(np.mean(np.square(audio))))
    dynamic_range = float(np.percentile(np.abs(audio), 95) - np.percentile(np.abs(audio), 10))
    clipping_ratio = float(np.mean(np.abs(audio) > 0.98))
    silence_ratio = float(np.mean(np.abs(audio) < 0.01))
    rms_norm = min(rms / 0.18, 1.0)
    dynamic_norm = min(dynamic_range / 0.35, 1.0)
    clipping_penalty = max(0.0, 1.0 - clipping_ratio * 8.0)
    silence_penalty = max(0.0, 1.0 - max(0.0, silence_ratio - 0.65))
    score = 100.0 * (0.35 * rms_norm + 0.30 * dynamic_norm + 0.20 * clipping_penalty + 0.15 * silence_penalty)
    return score, {
        "rms": rms,
        "dynamic_range": dynamic_range,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
    }


def audio_coherence_score(path: str | Path) -> tuple[float, dict[str, float]]:
    audio, sample_rate = load_wav_mono(path)
    if audio.size < sample_rate:
        return 50.0, {"window_count": 1.0, "rms_stability": 0.5, "centroid_stability": 0.5}
    window = sample_rate
    rms_values = []
    centroids = []
    for start in range(0, audio.size - window + 1, window):
        chunk = audio[start : start + window]
        rms_values.append(float(np.sqrt(np.mean(np.square(chunk))) + 1e-8))
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(chunk.size, d=1.0 / sample_rate)
        denom = float(np.sum(spectrum))
        centroid = float(np.sum(freqs * spectrum) / denom) if denom > 0 else 0.0
        centroids.append(centroid)
    rms_std = float(np.std(rms_values))
    centroid_std = float(np.std(centroids))
    rms_stability = max(0.0, 1.0 - min(1.0, rms_std / 0.12))
    centroid_stability = max(0.0, 1.0 - min(1.0, centroid_std / 2200.0))
    score = 100.0 * (0.55 * rms_stability + 0.45 * centroid_stability)
    return score, {
        "window_count": float(len(rms_values)),
        "rms_stability": rms_stability,
        "centroid_stability": centroid_stability,
    }


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return 0.0
    array_a = np.asarray(values_a, dtype=np.float32)
    array_b = np.asarray(values_b, dtype=np.float32)
    if np.std(array_a) < 1e-6 or np.std(array_b) < 1e-6:
        return 0.0
    return float(np.corrcoef(array_a, array_b)[0, 1])


def bounded_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def map_correlation_to_score(value: float) -> float:
    clipped = max(-1.0, min(1.0, value))
    return 50.0 * (clipped + 1.0)

