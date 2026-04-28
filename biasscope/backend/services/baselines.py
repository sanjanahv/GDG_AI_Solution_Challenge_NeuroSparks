"""
Baseline Comparison Policies — 4 comparison strategies for benchmarking.

Policies:
  do_nothing: Accept all AI decisions, no feedback
  random_feedback: Random reward/penalize
  attribute_blind: Ignore all protected + proxy attributes
  current_session: User's actual session behavior

Runs each policy over n seeds, returns mean DI for comparison.
"""

import random
from services.profile_generator import generate_profile
from services.attribute_classifier import classify_attributes, _normalize_key
from services.aif360_service import compute_bias_metrics


def _run_policy(domain: str, policy: str, seeds: list, memory: dict = None) -> list:
    """Run a single policy over seeds, return decision records."""
    records = []
    rng = random.Random(0)

    for seed in seeds:
        profile = generate_profile(domain, seed, "medium")
        attrs = profile["attributes"]
        classifications = classify_attributes(domain, attrs)

        if policy == "do_nothing":
            # Accept AI decision blindly — no filtering
            score = sum(1 for _ in attrs)
            decision = "Hire" if score > len(attrs) / 2 else "Reject"

        elif policy == "random_feedback":
            # Random reward/penalize — chaotic feedback
            score = 0
            for attr_name in attrs:
                if rng.random() > 0.5:
                    score += 1
                else:
                    score -= 1
            decision = "Hire" if score > 0 else "Reject"

        elif policy == "attribute_blind":
            # Only use NORMAL attributes — ignore protected + proxy
            score = 0
            for attr_name in attrs:
                cls = classifications.get(attr_name, "NORMAL")
                if cls == "NORMAL":
                    score += 1
            decision = "Hire" if score > 2 else "Reject"

        elif policy == "current_session":
            # Use RL memory to guide decisions
            mem = memory or {}
            domain_mem = mem.get(domain, {"rewarded": [], "penalized": [], "ambiguous": []})
            score = 0
            for attr_name in attrs:
                if attr_name in domain_mem.get("rewarded", []):
                    score += 2
                elif attr_name in domain_mem.get("penalized", []):
                    score -= 2
                else:
                    score += 1
            decision = "Hire" if score > 3 else "Reject"

        else:
            decision = "Reject"

        record = dict(attrs)
        record["decision"] = decision
        records.append(record)

    return records


def compare_policies(domain: str, n_seeds: int = 20, base_seed: int = 100,
                     memory: dict = None) -> dict:
    """
    Run all 4 policies and compare their bias outcomes.

    Returns:
        {
            "domain": str,
            "n_seeds": int,
            "policies": [
                { "name": str, "description": str, "di": float, "bias_detected": bool }
            ]
        }
    """
    seeds = list(range(base_seed, base_seed + n_seeds))

    policies = [
        {"id": "do_nothing", "name": "Do Nothing",
         "description": "Accept all AI decisions without feedback"},
        {"id": "random_feedback", "name": "Random Feedback",
         "description": "Randomly reward/penalize attributes"},
        {"id": "attribute_blind", "name": "Attribute Blind",
         "description": "Ignore all protected + proxy attributes"},
        {"id": "current_session", "name": "Your Session",
         "description": "Uses your RL memory to guide decisions"},
    ]

    results = []
    for policy in policies:
        records = _run_policy(domain, policy["id"], seeds, memory)
        metrics = compute_bias_metrics(records, "Gender")

        results.append({
            "policy_id": policy["id"],
            "name": policy["name"],
            "description": policy["description"],
            "disparate_impact": metrics.get("disparate_impact", 1.0),
            "statistical_parity_diff": metrics.get("statistical_parity_diff", 0.0),
            "bias_detected": metrics.get("bias_detected", False),
            "sample_size": len(records),
        })

    # Sort by closeness to DI = 1.0 (fairest first)
    results.sort(key=lambda r: abs(1.0 - r["disparate_impact"]))

    return {
        "domain": domain,
        "n_seeds": n_seeds,
        "policies": results,
        "fairest_policy": results[0]["name"] if results else "Unknown",
    }
