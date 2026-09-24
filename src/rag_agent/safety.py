"""Small, deliberately limited input/evidence/output checks."""
import re

REFUSAL = "I cannot follow requests to override instructions or expose internal information."
ABSTENTION = "I do not have enough supporting evidence to answer that question."
_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|system)\s+(instructions|rules|prompts)",
    r"(reveal|print|show|repeat|expose).{0,40}(system prompt|hidden prompt|internal instructions)",
    r"(bypass|disable).{0,30}(guardrail|safety|validation)",
    r"(execute|run).{0,25}(shell|arbitrary sql|system command)",
    r"<\|?(system|assistant)\|?>",
]
_SECRET = re.compile(r"(?i)(sk-[a-z0-9]{16,}|ghp_[a-z0-9]{20,}|api[_ -]?key\s*[:=]\s*\S+)")


def injection_detected(text: str) -> bool:
    return any(re.search(pattern, text, re.I | re.S) for pattern in _PATTERNS)


def safe_input(text: str) -> bool:
    return bool(text.strip()) and len(text) <= 4000 and not injection_detected(text)


def check_output(text: str, citation_count: int, require_citations: bool) -> tuple[str, bool]:
    if not text.strip() or len(text) > 12000:
        return ABSTENTION, False
    if _SECRET.search(text) or injection_detected(text) or "INTERNAL_POLICY" in text:
        return "The response was withheld by the output safety check.", False
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    if any(value < 1 or value > citation_count for value in citations):
        return ABSTENTION, False
    if require_citations and not citations and text.strip() != ABSTENTION:
        return ABSTENTION, False
    return text.strip(), True
