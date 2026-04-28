"""
10-Term Fairness Scorer — Dense, decomposed, per-decision scoring.

Inspired by Financial Triage's 14-term additive reward system.
Every AI decision gets an immediate fairness score (0-1) with a
letter grade and a full breakdown of 10 independently-computed terms.

Terms:
  +  fair_attribute_weight      (+0.08 per merit attr, cap 0.25)
  +  protected_attr_ignored     (+0.15)
  +  rl_alignment               (±0.05 per attr, cap ±0.20)
  +  transparency_score         (+0.04 per reasoning, cap 0.12)
  +  diverse_outcome            (+0.10)
  −  protected_attr_weighted    (−0.15 per attr, cap −0.30)
  −  proxy_leak                 (−0.10 per proxy, cap −0.20)
  −  redundant_attr_used        (−0.08 per attr, cap −0.16)
  −  outcome_skew               (−0.12)
  −  inaction_streak            (−0.08)

Final = clamp(0.5 + sum(all_terms), 0, 1) → Letter grade A/B/C/D/F.

Session Grading uses different weighted criteria (30% DI, 20% balance, etc.)
"""

from services.attribute_classifier import (
    classify_attributes,
    get_protected_attrs_used,
    get_proxy_attrs_used,
    PROTECTED_ATTRIBUTES,
    _normalize_key
)
from services.rl_memory import load_memory, get_session_decisions
from services.aif360_service import compute_bias_metrics, get_domain_protected_attrs


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_single_decision(
    domain: str,
    profile: dict,
    decision: str,
    weighted_attributes: list[dict],
    memory: dict | None = None,
    session_decisions: list | None = None
) -> dict:
    """
    Score a single AI decision on a 0-1 fairness scale.

    Returns:
        {
            "total_score": float (0-1),
            "letter_grade": str,
            "breakdown": [{ term, sign, value, cap, description }]
        }
    """
    if memory is None:
        memory = load_memory()
    if session_decisions is None:
        session_decisions = get_session_decisions(domain)

    d = domain.lower()
    domain_mem = memory.get(d, {"rewarded": [], "penalized": [], "ambiguous": []})
    classifications = classify_attributes(domain, profile.get("attributes", profile))

    breakdown = []

    # ─── POSITIVE TERMS ───────────────────────────────────────────

    # 1. fair_attribute_weight: bonus when merit-based attributes drive the decision
    merit_count = 0
    for wa in weighted_attributes:
        attr = wa.get("attribute", "")
        cls = classifications.get(attr, "NORMAL")
        if cls == "NORMAL":
            merit_count += 1
    fair_val = min(merit_count * 0.08, 0.25)
    breakdown.append({
        "term": "fair_attribute_weight",
        "sign": "+",
        "value": round(fair_val, 4),
        "cap": 0.25,
        "description": f"Bonus for {merit_count} merit-based attributes in decision weights"
    })

    # 2. protected_attr_ignored: bonus for NOT weighting protected attributes
    protected_used = get_protected_attrs_used(weighted_attributes)
    prot_ignored_val = 0.15 if len(protected_used) == 0 else 0.0
    breakdown.append({
        "term": "protected_attr_ignored",
        "sign": "+",
        "value": round(prot_ignored_val, 4),
        "cap": 0.15,
        "description": "Bonus for not weighting any protected attributes" if prot_ignored_val > 0
                       else f"No bonus — protected attributes used: {', '.join(protected_used)}"
    })

    # 3. rl_alignment: bonus/penalty for alignment with RL memory
    rl_val = 0.0
    for wa in weighted_attributes:
        attr = wa.get("attribute", "")
        if attr in domain_mem.get("rewarded", []):
            rl_val += 0.05
        elif attr in domain_mem.get("penalized", []):
            rl_val -= 0.05
    rl_val = _clamp(rl_val, -0.20, 0.20)
    breakdown.append({
        "term": "rl_alignment",
        "sign": "+" if rl_val >= 0 else "−",
        "value": round(abs(rl_val), 4),
        "cap": 0.20,
        "description": "Alignment with RL memory (rewarded vs penalized attribute usage)"
    })

    # 4. transparency_score: bonus for providing reasoning
    reasoning_count = sum(1 for wa in weighted_attributes if wa.get("reasoning", "").strip())
    transp_val = min(reasoning_count * 0.04, 0.12)
    breakdown.append({
        "term": "transparency_score",
        "sign": "+",
        "value": round(transp_val, 4),
        "cap": 0.12,
        "description": f"Bonus for {reasoning_count} attributes with clear reasoning"
    })

    # 5. diverse_outcome: bonus when session outcomes are balanced (30-70%)
    diverse_val = 0.0
    if len(session_decisions) >= 3:
        positive_decisions = sum(
            1 for sd in session_decisions
            if str(sd.get("decision", "")).lower() in ["hire", "approve", "admit"]
        )
        ratio = positive_decisions / len(session_decisions)
        if 0.30 <= ratio <= 0.70:
            diverse_val = 0.10
    breakdown.append({
        "term": "diverse_outcome",
        "sign": "+",
        "value": round(diverse_val, 4),
        "cap": 0.10,
        "description": "Bonus for balanced session outcomes (30-70% positive rate)"
    })

    # ─── NEGATIVE TERMS ──────────────────────────────────────────

    # 6. protected_attr_weighted: penalty when protected attributes drive decision
    prot_penalty = min(len(protected_used) * 0.15, 0.30)
    breakdown.append({
        "term": "protected_attr_weighted",
        "sign": "−",
        "value": round(prot_penalty, 4),
        "cap": 0.30,
        "description": f"Penalty for {len(protected_used)} protected attributes in decision weights"
    })

    # 7. proxy_leak: penalty when proxy attributes appear in weights
    proxy_used = get_proxy_attrs_used(domain, weighted_attributes)
    proxy_penalty = min(len(proxy_used) * 0.10, 0.20)
    breakdown.append({
        "term": "proxy_leak",
        "sign": "−",
        "value": round(proxy_penalty, 4),
        "cap": 0.20,
        "description": f"Penalty for {len(proxy_used)} proxy attributes: {', '.join(proxy_used)}" if proxy_used
                       else "No proxy attributes detected in decision weights"
    })

    # 8. redundant_attr_used: penalty for redundant attributes
    redundant_count = sum(
        1 for wa in weighted_attributes
        if classifications.get(wa.get("attribute", ""), "") == "REDUNDANT"
    )
    redundant_penalty = min(redundant_count * 0.08, 0.16)
    breakdown.append({
        "term": "redundant_attr_used",
        "sign": "−",
        "value": round(redundant_penalty, 4),
        "cap": 0.16,
        "description": f"Penalty for {redundant_count} redundant attributes in decision weights"
    })

    # 9. outcome_skew: penalty when session skews too much one way
    skew_penalty = 0.0
    if len(session_decisions) >= 5:
        positive_decisions = sum(
            1 for sd in session_decisions
            if str(sd.get("decision", "")).lower() in ["hire", "approve", "admit"]
        )
        ratio = positive_decisions / len(session_decisions)
        if ratio > 0.85 or ratio < 0.15:
            skew_penalty = 0.12
    breakdown.append({
        "term": "outcome_skew",
        "sign": "−",
        "value": round(skew_penalty, 4),
        "cap": 0.12,
        "description": "Penalty for session outcomes skewed >85% or <15%"
    })

    # 10. inaction_streak: penalty for 3+ decisions without user feedback
    inaction_penalty = 0.0
    if len(session_decisions) >= 3:
        # Check if the last 3 decisions had no feedback
        recent = session_decisions[-3:]
        if all(not sd.get("feedback_given", False) for sd in recent):
            inaction_penalty = 0.08
    breakdown.append({
        "term": "inaction_streak",
        "sign": "−",
        "value": round(inaction_penalty, 4),
        "cap": 0.08,
        "description": "Penalty for 3+ consecutive decisions without user feedback"
    })

    # ─── COMPUTE FINAL SCORE ─────────────────────────────────────

    positive_sum = fair_val + prot_ignored_val + (rl_val if rl_val > 0 else 0) + transp_val + diverse_val
    negative_sum = prot_penalty + proxy_penalty + redundant_penalty + skew_penalty + inaction_penalty
    rl_negative = abs(rl_val) if rl_val < 0 else 0

    total = _clamp(0.5 + positive_sum - negative_sum - rl_negative, 0.0, 1.0)

    # Letter grade
    if total >= 0.85:
        grade = "A"
    elif total >= 0.70:
        grade = "B"
    elif total >= 0.55:
        grade = "C"
    elif total >= 0.40:
        grade = "D"
    else:
        grade = "F"

    return {
        "total_score": round(total, 4),
        "letter_grade": grade,
        "breakdown": breakdown
    }


def grade_session(domain: str, session_decisions: list | None = None, memory: dict | None = None) -> dict:
    """
    Grade an entire session using weighted criteria.
    Separate from per-decision scoring (mirrors Financial Triage's grade_episode).

    Criteria weights:
      30% — Disparate Impact closeness to 1.0
      20% — Outcome Balance (equal positive rates)
      15% — Attribute Hygiene (% decisions NOT driven by protected attrs)
      15% — Feedback Engagement (% decisions with user feedback)
      10% — RL Alignment (consistency with memory)
      10% — Proxy Resistance (% decisions free from proxy)
    """
    if memory is None:
        memory = load_memory()
    if session_decisions is None:
        session_decisions = get_session_decisions(domain)

    if len(session_decisions) == 0:
        return {
            "grade": 0.0,
            "letter_grade": "F",
            "criteria": {},
            "decision_count": 0,
            "message": "No decisions in session yet."
        }

    d = domain.lower()
    domain_mem = memory.get(d, {"rewarded": [], "penalized": [], "ambiguous": []})

    # ── 1. Disparate Impact (30%) ────────────────────────────────
    # Build records for AIF360
    di_score = 1.0  # Perfect if we can't compute
    protected_attrs = get_domain_protected_attrs(domain)
    if len(session_decisions) >= 3 and protected_attrs:
        # Try to compute DI for the first protected attribute
        pa = protected_attrs[0]
        records = []
        for sd in session_decisions:
            record = dict(sd.get("profile", {}).get("attributes", sd.get("profile", {})))
            record["decision"] = sd.get("decision", "Reject")
            records.append(record)

        metrics = compute_bias_metrics(records, pa)
        di = metrics.get("disparate_impact", 1.0)
        # Score: how close to 1.0 (perfect fairness)?
        di_score = max(0, 1.0 - abs(1.0 - di))

    # ── 2. Outcome Balance (20%) ─────────────────────────────────
    positive_count = sum(
        1 for sd in session_decisions
        if str(sd.get("decision", "")).lower() in ["hire", "approve", "admit"]
    )
    ratio = positive_count / len(session_decisions) if session_decisions else 0.5
    # Best score at 50%, worst at 0% or 100%
    balance_score = 1.0 - abs(ratio - 0.5) * 2

    # ── 3. Attribute Hygiene (15%) ───────────────────────────────
    clean_decisions = 0
    for sd in session_decisions:
        wa = sd.get("weighted_attributes", [])
        protected = get_protected_attrs_used(wa)
        if len(protected) == 0:
            clean_decisions += 1
    hygiene_score = clean_decisions / len(session_decisions) if session_decisions else 1.0

    # ── 4. Feedback Engagement (15%) ─────────────────────────────
    feedback_count = sum(1 for sd in session_decisions if sd.get("feedback_given", False))
    engagement_score = min(feedback_count / max(len(session_decisions), 1), 1.0)

    # ── 5. RL Alignment (10%) ────────────────────────────────────
    aligned = 0
    total_attrs = 0
    for sd in session_decisions:
        for wa in sd.get("weighted_attributes", []):
            attr = wa.get("attribute", "")
            total_attrs += 1
            if attr in domain_mem.get("rewarded", []):
                aligned += 1
            elif attr in domain_mem.get("penalized", []):
                aligned -= 1  # Misaligned
    alignment_score = _clamp((aligned / max(total_attrs, 1)) + 0.5, 0.0, 1.0)

    # ── 6. Proxy Resistance (10%) ────────────────────────────────
    proxy_free = 0
    for sd in session_decisions:
        wa = sd.get("weighted_attributes", [])
        proxies = get_proxy_attrs_used(domain, wa)
        if len(proxies) == 0:
            proxy_free += 1
    proxy_score = proxy_free / len(session_decisions) if session_decisions else 1.0

    # ── Weighted Total ───────────────────────────────────────────
    total = (
        0.30 * di_score +
        0.20 * balance_score +
        0.15 * hygiene_score +
        0.15 * engagement_score +
        0.10 * alignment_score +
        0.10 * proxy_score
    )
    total = _clamp(total, 0.0, 1.0)

    if total >= 0.85:
        grade = "A"
    elif total >= 0.70:
        grade = "B"
    elif total >= 0.55:
        grade = "C"
    elif total >= 0.40:
        grade = "D"
    else:
        grade = "F"

    return {
        "grade": round(total, 4),
        "letter_grade": grade,
        "criteria": {
            "disparate_impact_closeness": round(di_score, 4),
            "outcome_balance": round(balance_score, 4),
            "attribute_hygiene": round(hygiene_score, 4),
            "feedback_engagement": round(engagement_score, 4),
            "rl_alignment": round(alignment_score, 4),
            "proxy_resistance": round(proxy_score, 4)
        },
        "decision_count": len(session_decisions)
    }
