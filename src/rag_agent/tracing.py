"""Allowlisted execution metadata; never log prompts, answers, IDs, or tool values."""

import json
import time
from pathlib import Path
from uuid import uuid4

_FIELDS = {"count", "route", "ok", "model", "citation_count", "reranked"}


class Trace:
    def __init__(self, directory: Path | None) -> None:
        self.run_id = uuid4().hex
        self.directory = directory
        self.started = time.monotonic()
        self.events: list[dict] = []

    def add(self, node: str, **metadata: object) -> None:
        self.events.append(
            {
                "node": node,
                "elapsed_ms": round((time.monotonic() - self.started) * 1000, 2),
                **{key: value for key, value in metadata.items() if key in _FIELDS},
            }
        )

    def save(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"{self.run_id}.json").write_text(
                json.dumps({"run_id": self.run_id, "events": self.events}, indent=2)
            )
