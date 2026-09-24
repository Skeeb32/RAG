"""Bounded in-process conversation history and explicit persistent preferences."""
import sqlite3
from collections import OrderedDict
from pathlib import Path

PREFERENCE_VALUES = {
    "verbosity": {"concise", "detailed"},
    "examples": {"python", "plain"},
}


class Memory:
    def __init__(self, path: Path, max_sessions: int = 100) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS preferences "
                       "(user TEXT, key TEXT, value TEXT, PRIMARY KEY(user, key))")

    def set_preference(self, user: str, key: str, value: str) -> None:
        if key not in PREFERENCE_VALUES or value not in PREFERENCE_VALUES[key]:
            raise ValueError("Unsupported preference key/value")
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?, ?)", (user, key, value))

    def preferences(self, user: str) -> dict[str, str]:
        with sqlite3.connect(self.path) as db:
            return dict(db.execute("SELECT key,value FROM preferences WHERE user=?", (user,)))

    def forget(self, user: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM preferences WHERE user=?", (user,))
        for key in list(self.sessions):
            if key[0] == user:
                del self.sessions[key]

    def history(self, user: str, session: str) -> list[dict[str, str]]:
        key = (user, session)
        if key in self.sessions:
            self.sessions.move_to_end(key)
        return [dict(message) for message in self.sessions.get(key, [])]

    def append(self, user: str, session: str, question: str, answer: str) -> None:
        key = (user, session)
        history = self.sessions.setdefault(key, [])
        history.extend([{"role": "user", "content": question[:4000]},
                        {"role": "assistant", "content": answer[:4000]}])
        self.sessions[key] = history[-6:]
        self.sessions.move_to_end(key)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
