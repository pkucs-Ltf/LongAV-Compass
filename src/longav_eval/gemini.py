from __future__ import annotations

import json
import os
import re
import mimetypes
from pathlib import Path
from typing import Any


class GeminiClient:
    def __init__(
        self,
        model: str,
        provider: str = "vertex",
        api_key: str | None = None,
        project: str | None = None,
        location: str = "global",
        credentials_json: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_key = api_key or ""
        self.project = project or ""
        self.location = location
        self.credentials_json = credentials_json or ""
        self._client = self._build_client()

    def score_media(
        self,
        prompt: str,
        media_paths: list[Path],
        mime_types: list[str],
    ) -> dict[str, Any]:
        from google.genai import types

        parts = []
        for media_path, mime_type in zip(media_paths, mime_types):
            parts.append(types.Part.from_bytes(data=media_path.read_bytes(), mime_type=mime_type))
        response = self._client.models.generate_content(
            model=self.model,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return parse_json_output(response.text or "{}")

    def _build_client(self):
        from google import genai

        if self.provider == "vertex":
            if not self.project:
                raise RuntimeError("Gemini vertex provider requires a non-empty project.")
            if not self.credentials_json:
                raise RuntimeError("Gemini vertex provider requires credentials_json.")
            credentials_path = Path(self.credentials_json).expanduser().resolve()
            if not credentials_path.exists():
                raise RuntimeError(f"Gemini credentials_json not found: {credentials_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
            return genai.Client(vertexai=True, location=self.location, project=self.project)

        if self.provider == "api_key":
            if not self.api_key:
                raise RuntimeError("Gemini api_key provider requires a non-empty api_key.")
            return genai.Client(api_key=self.api_key)

        raise RuntimeError(f"Unsupported Gemini provider '{self.provider}'")


def parse_json_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"
