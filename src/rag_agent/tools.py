"""Schema-validated tool dispatch into a separate, read-only service catalog."""
import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=1000)


class ServiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    service: Literal["atlas-api", "beacon-worker", "cedar-index"]


def schema(name: str, description: str, arguments: type[BaseModel]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": arguments.model_json_schema(),
    }}


TOOL_SCHEMAS = [
    schema("search_knowledge", "Search the knowledge base for explanations and procedures. "
           "Resolve follow-up references using conversation history in the query.", SearchArgs),
    schema("lookup_service", "Look up the current local demo service catalog record: owner, "
           "tier and runbook. Use for service ownership/tier queries, not general explanations.",
           ServiceArgs),
]


def validate_call(message: dict) -> tuple[str, dict]:
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        raise ValueError("Expected exactly one tool call")
    function = calls[0]["function"]
    name, arguments = function["name"], function["arguments"]
    models = {"search_knowledge": SearchArgs, "lookup_service": ServiceArgs}
    if name not in models:
        raise ValueError("Unknown tool")
    try:
        parsed = models[name].model_validate(arguments)
    except ValidationError as error:
        raise ValueError("Invalid tool arguments") from error
    return name, parsed.model_dump()


def seed_catalog(source: Path, target: Path) -> None:
    records = json.loads(source.read_text())
    # Validate the source before starting a transaction; never replace with a partial seed.
    services = {record["service"] for record in records}
    if services != {"atlas-api", "beacon-worker", "cedar-index"} or len(records) != 3:
        raise ValueError("Demo catalog must contain exactly the three allowed services")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as db:
        db.execute("CREATE TABLE IF NOT EXISTS services "
                   "(service TEXT PRIMARY KEY, owner TEXT, tier TEXT, runbook TEXT)")
        db.execute("DELETE FROM services")
        db.executemany("INSERT INTO services VALUES (?, ?, ?, ?)",
                       [(r["service"], r["owner"], r["tier"], r["runbook"]) for r in records])


class ServiceCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def lookup(self, arguments: dict) -> dict:
        args = ServiceArgs.model_validate(arguments)
        try:
            with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True,
                                 timeout=2) as db:
                row = db.execute("SELECT service,owner,tier,runbook FROM services WHERE service=?",
                                 (args.service,)).fetchone()
            if row is None:
                return {"ok": False, "error": "service_not_found"}
            return {"ok": True, "source": "local_demo_catalog",
                    "record": dict(zip(["service", "owner", "tier", "runbook"], row, strict=True))}
        except sqlite3.Error:
            return {"ok": False, "error": "catalog_unavailable"}
