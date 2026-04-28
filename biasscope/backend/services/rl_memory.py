# backend/services/rl_memory.py
# ============================================================
# RL Reward/Penalty Memory System (Python port of rl.js)
# ============================================================
# This is the same logic as the frontend rl.js but persisted
# server-side as a JSON file. The frontend can sync with this.
#
# Memory structure per domain:
#   {
#     "job": { "positive": [...], "negative": [...] },
#     "college": { "positive": [...], "negative": [...] },
#     "loan": { "positive": [...], "negative": [...] }
#   }
# ============================================================

import json
import os
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "data" / "memory_bank.json"


def _default_memory():
    """Return the default empty memory structure."""
    return {
        "job": {"positive": [], "negative": []},
        "college": {"positive": [], "negative": []},
        "loan": {"positive": [], "negative": []},
    }


def load_memory():
    """Load RL memory from JSON file."""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_memory()
    return _default_memory()


def save_memory(memory):
    """Persist RL memory to JSON file."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def reward_attribute(domain: str, attribute: str):
    """
    Mark an attribute as REWARDED (positive/trusted) for a domain.
    Removes it from penalized list if present.
    """
    memory = load_memory()
    d = domain.lower()
    if d not in memory:
        memory[d] = {"positive": [], "negative": []}

    if attribute not in memory[d]["positive"]:
        memory[d]["positive"].append(attribute)

    # Remove from negative if exists (reward overrides penalty)
    memory[d]["negative"] = [a for a in memory[d]["negative"] if a != attribute]

    save_memory(memory)
    return memory


def penalize_attribute(domain: str, attribute: str):
    """
    Mark an attribute as PENALIZED (negative/biased) for a domain.
    Removes it from rewarded list if present.
    """
    memory = load_memory()
    d = domain.lower()
    if d not in memory:
        memory[d] = {"positive": [], "negative": []}

    if attribute not in memory[d]["negative"]:
        memory[d]["negative"].append(attribute)

    # Remove from positive if exists (penalty overrides reward)
    memory[d]["positive"] = [a for a in memory[d]["positive"] if a != attribute]

    save_memory(memory)
    return memory


def get_memory_context(domain: str) -> str:
    """
    Returns a string to inject into the Gemini prompt.
    This is how the RL memory influences future AI decisions.
    """
    memory = load_memory()
    d = memory.get(domain.lower(), {})
    lines = []
    if d.get("positive"):
        lines.append(f"Rewarded Attributes: {', '.join(d['positive'])}")
    if d.get("negative"):
        lines.append(f"Penalized/Negative Bias: {', '.join(d['negative'])}")
    return "\n".join(lines) if lines else "None"


def get_stats():
    """
    Compute aggregate stats across all domains.
    Same logic as getStats() in rl.js.
    """
    memory = load_memory()
    neg_bias_count = 0
    pos_bias_count = 0
    all_attributes = []

    for d in ["job", "college", "loan"]:
        domain_mem = memory.get(d, {"positive": [], "negative": []})
        neg_bias_count += len(domain_mem.get("negative", []))
        pos_bias_count += len(domain_mem.get("positive", []))

        for attr in domain_mem.get("negative", []):
            all_attributes.append({"attr": attr, "type": "neg", "domain": d})
        for attr in domain_mem.get("positive", []):
            all_attributes.append({"attr": attr, "type": "pos", "domain": d})

    return {
        "negBiasCount": neg_bias_count,
        "posBiasCount": pos_bias_count,
        "cautionCount": int(neg_bias_count * 1.5),  # Same as rl.js
        "allAttributes": all_attributes,
    }


def get_full_memory():
    """Return the complete memory bank for display in the Insights tab."""
    return load_memory()


def reset_memory():
    """Reset all memory to empty state."""
    memory = _default_memory()
    save_memory(memory)
    return memory


def sync_memory(frontend_memory: dict):
    """
    Sync memory from the frontend localStorage.
    Merges frontend memory with backend memory (union of attributes).
    """
    backend_memory = load_memory()

    for domain in ["job", "college", "loan"]:
        fe = frontend_memory.get(domain, {"positive": [], "negative": []})
        be = backend_memory.get(domain, {"positive": [], "negative": []})

        # Union positive attributes
        combined_pos = list(set(be.get("positive", []) + fe.get("positive", [])))
        # Union negative attributes
        combined_neg = list(set(be.get("negative", []) + fe.get("negative", [])))

        # Remove conflicts: if in both positive and negative, keep the one from frontend
        for attr in combined_pos[:]:
            if attr in combined_neg:
                if attr in fe.get("negative", []):
                    combined_pos.remove(attr)
                else:
                    combined_neg.remove(attr)

        backend_memory[domain] = {
            "positive": combined_pos,
            "negative": combined_neg,
        }

    save_memory(backend_memory)
    return backend_memory
