"""Command-line entry points for ingestion, retrieval, chat, and preferences."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_agent.agent import Agent
from rag_agent.config import Settings
from rag_agent.llm import OllamaModel
from rag_agent.memory import Memory
from rag_agent.retrieval import (
    CrossEncoderReranker,
    HybridIndex,
    SentenceEncoder,
    chunk_documents,
)
from rag_agent.tools import ServiceCatalog, seed_catalog


def load_index(settings: Settings) -> HybridIndex:
    if not (settings.index / "index.npz").is_file():
        raise ValueError("Index missing. Run rag-agent ingest first.")
    encoder = SentenceEncoder(settings.embedding_model)
    return HybridIndex.load(settings.index, encoder, settings.embedding_model)


def make_agent(settings: Settings, rerank: bool = True) -> Agent:
    return Agent(load_index(settings),
                 CrossEncoderReranker(settings.reranker_model) if rerank else None,
                 OllamaModel(settings.ollama_url, settings.model),
                 ServiceCatalog(settings.index / "catalog.sqlite3"),
                 Memory(settings.memory), settings.traces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--docs", type=Path, default=Path("data/sample"))
    ingest.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    ingest.add_argument("--chunk-size", type=int, default=160)
    ingest.add_argument("--overlap", type=int, default=30)
    search = commands.add_parser("search")
    search.add_argument("question")
    search.add_argument("--baseline", action="store_true")
    for command in ("ask", "chat"):
        sub = commands.add_parser(command)
        if command == "ask":
            sub.add_argument("question")
        sub.add_argument("--user", default="local")
        sub.add_argument("--session", default="default")
        sub.add_argument("--baseline", action="store_true")
    preference = commands.add_parser("preference")
    preference.add_argument("key", choices=["verbosity", "examples"])
    preference.add_argument("value")
    preference.add_argument("--user", default="local")
    forget = commands.add_parser("forget")
    forget.add_argument("--user", default="local")
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        if args.command == "ingest":
            chunks = chunk_documents(args.docs, args.chunk_size, args.overlap)
            encoder = SentenceEncoder(settings.embedding_model)
            index = HybridIndex(chunks, encoder.encode([c.text for c in chunks]), encoder)
            fingerprint = index.save(settings.index, settings.embedding_model,
                                     args.chunk_size, args.overlap)
            seed_catalog(args.catalog, settings.index / "catalog.sqlite3")
            print(json.dumps({"chunks": len(chunks), "fingerprint": fingerprint}))
        elif args.command == "search":
            hits = load_index(settings).search(args.question)
            if not args.baseline:
                hits = CrossEncoderReranker(settings.reranker_model).rank(args.question, hits, 3)
            print(json.dumps([asdict(hit) for hit in hits[:3]], indent=2))
        elif args.command in {"preference", "forget"}:
            memory = Memory(settings.memory)
            if args.command == "forget":
                memory.forget(args.user)
            else:
                memory.set_preference(args.user, args.key, args.value)
            print("Preferences updated.")
        else:
            agent = make_agent(settings, rerank=not args.baseline)
            if args.command == "ask":
                print(json.dumps(agent.ask(args.question, args.user, args.session), indent=2))
            else:
                print("Ask about the knowledge base or service catalog. /quit exits; /forget clears memory.")
                while True:
                    try:
                        question = input("> ")
                    except EOFError:
                        break
                    if question == "/quit":
                        break
                    if question == "/forget":
                        agent.memory.forget(args.user)
                        print("Memory cleared.")
                        continue
                    result = agent.ask(question, args.user, args.session)
                    print(result["answer"])
                    print(f"[{result['status']}; trace {result['run_id']}]")
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
