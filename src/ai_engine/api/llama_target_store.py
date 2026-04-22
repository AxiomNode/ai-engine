from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PersistedLlamaTarget:
    url: str
    label: str | None
    updated_at: str


class LlamaTargetStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)

    async def load(self) -> PersistedLlamaTarget | None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

        if not isinstance(payload, dict):
            return None

        url = payload.get("url")
        updated_at = payload.get("updatedAt")
        label = payload.get("label")
        if not isinstance(url, str) or not isinstance(updated_at, str):
            return None

        return PersistedLlamaTarget(
            url=url,
            label=label if isinstance(label, str) and label.strip() else None,
            updated_at=updated_at,
        )

    async def save(self, target: PersistedLlamaTarget) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "url": target.url,
            "label": target.label,
            "updatedAt": target.updated_at,
        }
        self._path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    async def reset(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            return
