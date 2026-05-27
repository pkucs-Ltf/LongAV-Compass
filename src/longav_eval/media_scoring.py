from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .media import run_command, video_encoder_args


def analyze_boundary_technical(video_path: Path, fps: int = 8) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps},scale=160:-1",
                str(frames_dir / "frame_%04d.jpg"),
            ]
        )
        frames = [_load_frame(path) for path in sorted(frames_dir.glob("frame_*.jpg"))]

    if not frames:
        return {
            "black_frame_ratio": 1.0,
            "flash_count": 0,
            "duplicate_frame_ratio": 0.0,
            "freeze_max_sec": 0.0,
            "technical_boundary_score": 0.0,
        }

    brightness = [float(frame.mean()) for frame in frames]
    black_frame_ratio = sum(value < 0.05 for value in brightness) / len(brightness)
    diffs = [float(np.mean(np.abs(frames[index] - frames[index - 1]))) for index in range(1, len(frames))]
    duplicate_flags = [diff < 0.01 for diff in diffs]
    duplicate_frame_ratio = (sum(duplicate_flags) / len(duplicate_flags)) if duplicate_flags else 0.0
    freeze_max_sec = _max_true_run(duplicate_flags) / fps
    flash_count = sum(1 for index in range(1, len(brightness)) if abs(brightness[index] - brightness[index - 1]) > 0.45)

    penalty = (
        70.0 * black_frame_ratio
        + 45.0 * duplicate_frame_ratio
        + 12.0 * freeze_max_sec
        + 8.0 * flash_count
    )
    technical_score = round(max(0.0, min(100.0, 100.0 - penalty)), 4)
    return {
        "black_frame_ratio": round(black_frame_ratio, 4),
        "flash_count": flash_count,
        "duplicate_frame_ratio": round(duplicate_frame_ratio, 4),
        "freeze_max_sec": round(freeze_max_sec, 4),
        "technical_boundary_score": technical_score,
    }


def _make_preview(source: Path, output: Path, fps: int = 4, max_width: int = 512) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    scale_filter = f"fps={fps},scale='min({max_width},iw)':-2"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            scale_filter,
            *video_encoder_args(crf=32),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    return output


def _score_video_json(client, prompt: str, video_path: Path) -> dict[str, Any]:
    last_payload: dict[str, Any] = {}
    for attempt in range(3):
        try:
            payload = client.score_media(prompt, [video_path], ["video/mp4"])
        except json.JSONDecodeError:
            payload = {}
        if payload:
            return payload
        last_payload = payload
        if attempt < 2:
            time.sleep(2.0 * (attempt + 1))
    return last_payload


def _load_frame(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _max_true_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
