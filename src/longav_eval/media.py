from __future__ import annotations

import json
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .runtime import StreamInfo
from .schemas import ConfigError


@lru_cache(maxsize=None)
def _ffmpeg_encoders(binary: str = "ffmpeg") -> frozenset[str]:
    try:
        completed = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    encoders: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in {"V", "A", "S"}:
            encoders.add(parts[1])
    return frozenset(encoders)


def video_encoder_args(crf: int = 30, binary: str = "ffmpeg") -> list[str]:
    encoders = _ffmpeg_encoders(binary)
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    if "libopenh264" in encoders:
        bitrate = "4M" if crf <= 22 else "2M" if crf <= 30 else "1M"
        return ["-c:v", "libopenh264", "-b:v", bitrate]
    if "mpeg4" in encoders:
        qscale = "3" if crf <= 22 else "5" if crf <= 30 else "7"
        return ["-c:v", "mpeg4", "-q:v", qscale]
    return ["-c:v", "mpeg4", "-q:v", "5"]


def run_command(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(part) for part in args)
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Command failed: {command}\n{message}") from exc


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"ffprobe failed for {path}: {message}") from exc
    return json.loads(completed.stdout)


def _parse_fps(raw_value: str | None) -> float | None:
    if not raw_value or raw_value == "0/0":
        return None
    if "/" in raw_value:
        numerator, denominator = raw_value.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(raw_value)
    except ValueError:
        return None


def probe_video(path: str | Path) -> StreamInfo | None:
    data = ffprobe_json(path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        duration = stream.get("duration") or data.get("format", {}).get("duration")
        return StreamInfo(
            codec=stream.get("codec_name"),
            duration_sec=float(duration) if duration else None,
            fps=_parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
            width=int(stream["width"]) if stream.get("width") else None,
            height=int(stream["height"]) if stream.get("height") else None,
        )
    return None


def probe_audio(path: str | Path) -> StreamInfo | None:
    data = ffprobe_json(path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        duration = stream.get("duration") or data.get("format", {}).get("duration")
        return StreamInfo(
            codec=stream.get("codec_name"),
            duration_sec=float(duration) if duration else None,
            sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            channels=int(stream["channels"]) if stream.get("channels") else None,
        )
    return None


def ensure_exists(path: Path | None, label: str) -> Path:
    if path is None:
        raise ConfigError(f"Missing required path for {label}")
    if not path.exists():
        raise ConfigError(f"{label} does not exist: {path}")
    return path


def normalize_video(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-an",
            *video_encoder_args(crf=20),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def normalize_audio(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def extract_segment_video(source: Path, output: Path, start_sec: float, end_sec: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            str(source),
            "-an",
            *video_encoder_args(crf=20),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def extract_segment_audio(source: Path, output: Path, start_sec: float, end_sec: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def extract_frame(source: Path, output: Path, timestamp_sec: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_sec:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(output),
        ]
    )

