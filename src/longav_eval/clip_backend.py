from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_MODEL = "xlm-roberta-base-ViT-B-32"
DEFAULT_PRETRAINED = "laion5b_s13b_b90k"
DEFAULT_OPENAI_CLIP_MODEL = "ViT-B/32"


class ClipBackendError(RuntimeError):
    """Raised when the optional CLIP backend is unavailable or cannot score inputs."""


def text_video_alignment_score(text: str, frame_paths: Iterable[str | Path]) -> dict[str, Any]:
    paths = _valid_paths(frame_paths)
    if not text.strip():
        raise ClipBackendError("Missing text for CLIP text-video alignment.")
    if not paths:
        raise ClipBackendError("Missing sampled video frames for CLIP text-video alignment.")

    text_chunks = _text_chunks(text)
    text_features = _encode_texts(text_chunks)
    image_features = _encode_images(paths)
    cosine_matrix = _cosine(image_features, text_features)
    cosine = np.max(cosine_matrix, axis=1)
    per_frame = [_text_clipscore(value) for value in cosine.tolist()]
    score = _weighted_frame_score(per_frame)
    payload = _score_payload(
        score=score,
        cosine=cosine,
        frame_paths=paths,
        score_key="text_video_alignment_clip",
        score_method="clipscore_text_image_250x",
    )
    payload["text_chunk_count"] = len(text_chunks)
    return payload


def image_video_alignment_score(reference_image_path: str | Path, frame_paths: Iterable[str | Path]) -> dict[str, Any]:
    paths = _valid_paths(frame_paths)
    reference = Path(reference_image_path)
    if not reference.exists():
        raise ClipBackendError(f"Missing reference image for CLIP image-video alignment: {reference}")
    if not paths:
        raise ClipBackendError("Missing sampled video frames for CLIP image-video alignment.")

    reference_features = _encode_images([reference])
    image_features = _encode_images(paths)
    cosine = _cosine(image_features, reference_features)[:, 0]
    per_frame = [_image_clipscore(value) for value in cosine.tolist()]
    score = _weighted_frame_score(per_frame)
    return _score_payload(
        score=score,
        cosine=cosine,
        frame_paths=paths,
        score_key="image_video_alignment_clip",
        score_method="clip_image_image_cosine",
    )


def image_semantic_alignment_score(
    reference_image_path: str | Path,
    text: str,
    frame_paths: Iterable[str | Path],
    image_weight: float = 0.65,
) -> dict[str, Any]:
    image_payload = image_video_alignment_score(reference_image_path, frame_paths)
    text_payload = text_video_alignment_score(text, frame_paths)
    text_weight = 1.0 - image_weight
    score = image_weight * float(image_payload["image_video_alignment_clip"]) + text_weight * float(
        text_payload["text_video_alignment_clip"]
    )
    return {
        "backend": "clip",
        "clip_model": image_payload["clip_model"],
        "clip_pretrained": image_payload["clip_pretrained"],
        "clip_device": image_payload["clip_device"],
        "score_method": f"{image_weight:.2f}*image_video + {text_weight:.2f}*text_video",
        "image_semantic_alignment_clip": round(score, 6),
        "image_video_alignment_clip": image_payload["image_video_alignment_clip"],
        "text_video_alignment_clip": text_payload["text_video_alignment_clip"],
        "image_mean_cosine": image_payload["mean_cosine"],
        "text_mean_cosine": text_payload["mean_cosine"],
        "frame_count": image_payload["frame_count"],
    }


def clip_model_info() -> dict[str, str]:
    backend, model_name, pretrained, device = _clip_config()
    return {"clip_backend": backend, "clip_model": model_name, "clip_pretrained": pretrained, "clip_device": device}


def _score_payload(
    score: float,
    cosine: np.ndarray,
    frame_paths: list[Path],
    score_key: str,
    score_method: str,
) -> dict[str, Any]:
    info = _runtime_clip_model_info()
    values = cosine.tolist()
    payload: dict[str, Any] = {
        "backend": "clip",
        **info,
        "score_method": score_method,
        score_key: round(float(score), 6),
        "mean_cosine": round(float(np.mean(cosine)), 6),
        "min_cosine": round(float(np.min(cosine)), 6),
        "max_cosine": round(float(np.max(cosine)), 6),
        "frame_count": len(frame_paths),
        "frame_paths": [path.as_posix() for path in frame_paths],
        "frame_cosines": [round(float(value), 6) for value in values],
    }
    return payload


def _weighted_frame_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    # Mean keeps the metric stable; the small lower-tail term penalizes isolated off-prompt segments.
    return float(0.8 * np.mean(arr) + 0.2 * np.percentile(arr, 25))


def _text_clipscore(cosine: float) -> float:
    return float(np.clip(2.5 * max(float(cosine), 0.0), 0.0, 1.0))


def _image_clipscore(cosine: float) -> float:
    return float(np.clip(float(cosine), 0.0, 1.0))


def _valid_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths if path and Path(path).exists()]


def _text_chunks(text: str, max_chars: int = 220, max_chunks: int = 16) -> list[str]:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    chunks: list[str] = []
    for separator in ("\n", "。", ".", ";", "；"):
        if separator in cleaned:
            parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
            break
    else:
        parts = [cleaned]
    for part in parts:
        while len(part) > max_chars:
            chunks.append(part[:max_chars].strip())
            part = part[max_chars:].strip()
        if part:
            chunks.append(part)
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks] or [cleaned[:max_chars]]


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.matmul(left, right.T)


def _encode_images(paths: list[Path]) -> np.ndarray:
    import torch

    model, _, preprocess, _, device = _load_clip()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(preprocess(image.convert("RGB")))
    batch = torch.stack(images).to(device)
    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().float().numpy()


def _encode_texts(texts: list[str]) -> np.ndarray:
    import torch

    model, tokenizer, _, _, device = _load_clip()
    tokens = tokenizer(texts).to(device)
    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().float().numpy()


@lru_cache(maxsize=1)
def _load_clip():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise ClipBackendError("PyTorch is required for CLIP scoring. Install with `pip install -e .[clip]`.") from exc

    backend, model_name, pretrained, device_name = _clip_config()
    errors: list[str] = []

    if backend in {"auto", "open_clip"}:
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device_name,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            info = {
                "clip_backend": "open_clip",
                "clip_model": model_name,
                "clip_pretrained": pretrained,
                "clip_device": device_name,
            }
            return model, tokenizer, preprocess, info, torch.device(device_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"open_clip: {exc}")
            if backend == "open_clip":
                raise ClipBackendError(
                    f"Failed to load OpenCLIP model '{model_name}' with pretrained='{pretrained}'."
                ) from exc

    if backend in {"auto", "openai_clip"}:
        openai_model = os.getenv("LONGAV_OPENAI_CLIP_MODEL", DEFAULT_OPENAI_CLIP_MODEL)
        try:
            import clip

            model, preprocess = clip.load(openai_model, device=device_name)
            tokenizer = lambda texts: clip.tokenize(texts, truncate=True)
            model.eval()
            info = {
                "clip_backend": "openai_clip",
                "clip_model": openai_model,
                "clip_pretrained": "openai",
                "clip_device": device_name,
            }
            return model, tokenizer, preprocess, info, torch.device(device_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openai_clip: {exc}")
            if backend == "openai_clip":
                raise ClipBackendError(f"Failed to load OpenAI CLIP model '{openai_model}'.") from exc

    details = "; ".join(errors) if errors else "no backend attempted"
    raise ClipBackendError(f"CLIP backend is not installed or failed to load ({details}). Install with `pip install -e .[clip]`.")


def _runtime_clip_model_info() -> dict[str, str]:
    _, _, _, info, _ = _load_clip()
    return info


def _clip_config() -> tuple[str, str, str, str]:
    backend = os.getenv("LONGAV_CLIP_BACKEND", "auto")
    model_name = os.getenv("LONGAV_CLIP_MODEL", DEFAULT_MODEL)
    pretrained = os.getenv("LONGAV_CLIP_PRETRAINED", DEFAULT_PRETRAINED)
    device = os.getenv("LONGAV_CLIP_DEVICE")
    if not device:
        try:
            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    return backend, model_name, pretrained, device
