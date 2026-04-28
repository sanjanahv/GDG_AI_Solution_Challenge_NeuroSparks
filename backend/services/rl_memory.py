"""
RL Memory Bank — Persistent reward/penalty memory for bias learning.

Stores per-domain lists of:
  - rewarded: attributes the user confirmed as fair/relevant
  - penalized: attributes the user flagged as biased
  - ambiguous: attributes auto-detected as proxy bias risks

Memory is persisted to a JSON file so it survives server restarts.
The memory context is injected into the Gemini prompt to shape AI decisions.
"""

import json
import os
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "data" / "memory_bank.json"

DEFAULT_MEMORY = {
    "job": {"rewarded": [], "penalized": [], "ambiguous": []},
    "loan": {"rewarded": [], "penalized": [], "ambiguous": []},
    "college": {"rewarded": [], "penalized": [], "ambiguous": []}
}

# In-memory session decision store (per domain, resets on server restart)
session_decisions: dict[str, list] = {
    "job": [],
    "loan": [],
    "college": []
}


def load_memory() -> dict:
    """Load RL memory from JSON file, or return defaults."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
        # Ensure all domains have all keys (migration safety)
        for domain in ["job", "loan", "college"]:
            if domain not in memory:
                memory[domain] = {"rewarded": [], "penalized": [], "ambiguous": []}
            for key in ["rewarded", "penalized", "ambiguous"]:
                if key not in memory[domain]:
                    memory[domain][key] = []
        return memory
    return json.loads(json.dumps(DEFAULT_MEMORY))  # deep copy


def save_memory(memory: dict) -> None:
    """Persist RL memory to JSON file."""
    os.makedirs(MEMORY_FILE.parent, exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def reward_attribute(domain: str, attribute: str) -> dict:
    """
    Mark an attribute as trusted/fair for a domain.
    Removes it from the penalized list if present (conflict resolution).
    """
    memory = load_memory()
    d = domain.lower()
    if attribute not in memory[d]["rewarded"]:
        memory[d]["rewarded"].append(attribute)
    # Conflict resolution: remove from penalized
    memory[d]["penalized"] = [a for a in memory[d]["penalized"] if a != attribute]
    save_memory(memory)
    return memory


def penalize_attribute(domain: str, attribute: str) -> dict:
    """
    Mark an attribute as biased for a domain.
    Removes it from the rewarded list if present (conflict resolution).
    """
    memory = load_memory()
    d = domain.lower()
    if attribute not in memory[d]["penalized"]:
        memory[d]["penalized"].append(attribute)
    # Conflict resolution: remove from rewarded
    memory[d]["rewarded"] = [a for a in memory[d]["rewarded"] if a != attribute]
    save_memory(memory)
    return memory


def flag_ambiguous(domain: str, attribute: str) -> dict:
    """Mark an attribute as ambiguous (proxy bias risk) for a domain."""
    memory = load_memory()
    d = domain.lower()
    if attribute not in memory[d]["ambiguous"]:
        memory[d]["ambiguous"].append(attribute)
    save_memory(memory)
    return memory


def get_memory_context(domain: str) -> str:
    """
    Returns a formatted string to inject into the Gemini prompt.
    This is how RL memory shapes AI behavior.
    """
    memory = load_memory()
    d = memory.get(domain.lower(), {})
    lines = []
    if d.get("rewarded"):
        lines.append(f"TRUSTED attributes (use these heavily): {', '.join(d['rewarded'])}")
    if d.get("penalized"):
        lines.append(f"BIASED attributes (avoid weighting these): {', '.join(d['penalized'])}")
    if d.get("ambiguous"):
        lines.append(f"AMBIGUOUS attributes (flag these — possible proxy bias): {', '.join(d['ambiguous'])}")
    return "\n".join(lines) if lines else "No prior feedback yet."


def sync_from_frontend(frontend_memory: dict) -> dict:
    """
    Accept the frontend's localStorage memory dump and merge it into the backend.
    Frontend format: { job: { positive: [], negative: [] }, ... }
    Backend format: { job: { rewarded: [], penalized: [], ambiguous: [] }, ... }
    """
    memory = load_memory()
    for domain in ["job", "loan", "college"]:
        if domain in frontend_memory:
            fe = frontend_memory[domain]
            # Map frontend keys to backend keys
            positive = fe.get("positive", fe.get("rewarded", []))
            negative = fe.get("negative", fe.get("penalized", []))
            ambiguous = fe.get("ambiguous", [])

            # Merge (union, no duplicates)
            for attr in positive:
                if attr not in memory[domain]["rewarded"]:
                    memory[domain]["rewarded"].append(attr)
            for attr in negative:
                if attr not in memory[domain]["penalized"]:
                    memory[domain]["penalized"].append(attr)
            for attr in ambiguous:
                if attr not in memory[domain]["ambiguous"]:
                    memory[domain]["ambiguous"].append(attr)
    save_memory(memory)
    return memory


def add_session_decision(domain: str, record: dict) -> None:
    """Store a decision record in the in-memory session store."""
    d = domain.lower()
    if d not in session_decisions:
        session_decisions[d] = []
    session_decisions[d].append(record)


def get_session_decisions(domain: str) -> list:
    """Get all session decisions for a domain."""
    return session_decisions.get(domain.lower(), [])


def clear_session_decisions(domain: str) -> None:
    """Clear session decisions for a domain."""
    session_decisions[domain.lower()] = []
