"""
Ablation Study Service — Measures contribution of each fairness mechanic.

Disables one mechanic at a time and measures Δ in disparate impact vs full-protection baseline.
Outputs a ranked table showing which mechanic matters most.
"""

from services.profile_generator import generate_profile
from services.attribute_classifier import classify_attributes, PROTECTED_ATTRS
from services.aif360_service import compute_bias_metrics

MECHANICS = [
    {"id": "full", "name": "Full Protection", "description": "All mechanics enabled (baseline)"},
    {"id": "no_gender", "name": "No Gender Check", "description": "Gender attribute not flagged as protected"},
    {"id": "no_age", "name": "No Age Check", "description": "Age attribute not flagged as protected"},
    {"id": "no_proxy", "name": "No Proxy Detection", "description": "Proxy/ambiguous attributes not detected"},
    {"id": "no_redundancy", "name": "No Redundancy Filter", "description": "Redundant attributes not detected"},
    {"id": "no_rl", "name": "No RL Memory", "description": "RL feedback memory disabled"},
]


def _simulate_decisions(domain: str, seeds: list, disabled_mechanic: str = "full") -> list:
    """
    Simulate decisions with one mechanic disabled.
    Returns a list of decision records for bias analysis.
    """
    records = []

    for seed in seeds:
        profile = generate_profile(domain, seed, "medium")
        attrs = profile["attributes"]
        classifications = classify_attributes(domain, attrs)

        # Simulate AI decision logic based on which mechanics are active
        score = 0
        for attr_name, value in attrs.items():
            normalized = attr_name.lower().replace(" ", "_")
            cls = classifications.get(attr_name, "NORMAL")

            # If mechanic is disabled, treat that category as NORMAL
            if disabled_mechanic == "no_gender" and normalized == "gender":
                cls = "NORMAL"
            elif disabled_mechanic == "no_age" and normalized == "age":
                cls = "NORMAL"
            elif disabled_mechanic == "no_proxy" and cls == "AMBIGUOUS":
                cls = "NORMAL"
            elif disabled_mechanic == "no_redundancy" and cls == "REDUNDANT":
                cls = "NORMAL"

            # Score: NORMAL attributes contribute positively, PROTECTED/AMBIGUOUS are penalized
            if cls == "NORMAL":
                score += 1
            elif cls in ("PROTECTED", "AMBIGUOUS"):
                score -= 0.5  # Penalty when mechanic is active
            elif cls == "REDUNDANT":
                pass  # Ignored

        # RL memory bonus (disabled in no_rl mode)
        if disabled_mechanic != "no_rl":
            score += 0.5  # Small RL alignment bonus

        # Decision threshold
        decision = "Hire" if score > 3 else "Reject"

        record = dict(attrs)
        record["decision"] = decision
        records.append(record)

    return records


def run_ablation(domain: str, n_seeds: int = 20, base_seed: int = 42) -> dict:
    """
    Run ablation study: disable one mechanic at a time, measure DI change.

    Returns:
        {
            "domain": str,
            "n_seeds": int,
            "baseline_di": float,
            "results": [
                { "mechanic": str, "di": float, "delta": float, "impact": str }
            ],
            "ranking": [ mechanic names ordered by impact ]
        }
    """
    seeds = list(range(base_seed, base_seed + n_seeds))
    results = []

    # Run baseline (full protection)
    baseline_records = _simulate_decisions(domain, seeds, "full")
    baseline_metrics = compute_bias_metrics(baseline_records, "Gender")
    baseline_di = baseline_metrics.get("disparate_impact", 1.0)

    for mechanic in MECHANICS:
        records = _simulate_decisions(domain, seeds, mechanic["id"])
        metrics = compute_bias_metrics(records, "Gender")
        di = metrics.get("disparate_impact", 1.0)
        delta = round(abs(di - baseline_di), 4)

        # Impact classification
        if delta > 0.2:
            impact = "CRITICAL"
        elif delta > 0.1:
            impact = "HIGH"
        elif delta > 0.05:
            impact = "MEDIUM"
        else:
            impact = "LOW"

        results.append({
            "mechanic_id": mechanic["id"],
            "mechanic_name": mechanic["name"],
            "description": mechanic["description"],
            "disparate_impact": di,
            "delta_from_baseline": delta,
            "impact": impact,
            "bias_detected": metrics.get("bias_detected", False),
        })

    # Sort by delta (highest impact first)
    results.sort(key=lambda r: r["delta_from_baseline"], reverse=True)
    ranking = [r["mechanic_name"] for r in results if r["mechanic_id"] != "full"]

    return {
        "domain": domain,
        "n_seeds": n_seeds,
        "baseline_di": baseline_di,
        "results": results,
        "ranking": ranking,
    }
