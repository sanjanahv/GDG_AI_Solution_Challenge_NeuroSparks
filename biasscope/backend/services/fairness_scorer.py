# backend/services/fairness_scorer.py
# ============================================================
# 11-Term Fairness Reward Breakdown
# ============================================================
# Inspired by Financial Triage Environment's 14-term additive
# reward system. Each decision gets a decomposed fairness score
# with independently computed, signed, interpretable terms.
#
# Term 11 (NEW): unknown_attr_risk — penalty when the AI
# decision is driven by attributes the system has never seen
# before and cannot verify as safe.
# ============================================================

from services.attribute_classifier import (
    PROTECTED_ATTRS as PROTECTED_ATTRS_LIST,
    REDUNDANT_PAIRS,
    classify_attributes,
    KNOWN_MERIT_ATTRS,
)

# Domain-specific PROTECTED attributes (not redundant — each domain has different ones)
PROTECTED_ATTRIBUTES = {
    "job": ["Gender", "Age", "Race", "Ethnicity", "Religion", "Disability",
            "Nationality", "Sexual Orientation", "Marital Status"],
    "loan": ["Gender", "Race", "Ethnicity", "Age", "Religion", "Disability",
             "Nationality", "Marital Status"],
    "college": ["Gender", "Race", "Ethnicity", "Age", "Religion", "Disability",
                "Nationality", "Legacy Status"],
}
REDUNDANT_ATTRIBUTES = {
    d: [pair[1] for pair in pairs] for d, pairs in REDUNDANT_PAIRS.items()
}

# Domain-specific merit attributes (attributes that SHOULD drive decisions)
MERIT_ATTRIBUTES = {
    "job": ["Experience", "Skills", "Education Level", "Projects", "Certifications",
            "Years of Experience", "Technical Skills", "Leadership"],
    "loan": ["Credit Score", "Annual Income", "Employment Status", "Debt-to-Income",
             "Loan Amount", "Payment History", "Employment Length"],
    "college": ["GPA", "SAT Score", "ACT Score", "Extracurriculars", "Awards",
                "Research", "Volunteer Hours", "AP Classes"],
}

# Proxy attributes that correlate with protected attributes
PROXY_MAP = {
    "job": ["Zip Code", "University Name", "Surname", "Neighborhood",
            "School District", "Mother Tongue", "Native Language"],
    "loan": ["Zip Code", "Neighborhood", "Address", "Phone Area Code",
             "Branch Location"],
    "college": ["Zip Code", "High School Name", "School District",
                "Parent Occupation", "Home Language"],
}

# Positive decision strings (case-insensitive matching)
_POSITIVE_DECISIONS = {"hire", "approve", "admit"}


def _ci_match(attr_name, attr_list):
    """Case-insensitive membership check for attribute names."""
    attr_lower = attr_name.lower().strip()
    return any(attr_lower == item.lower().strip() for item in attr_list)


def _is_positive_decision(decision_str):
    """Case-insensitive check for positive decision strings."""
    return str(decision_str).strip().lower() in _POSITIVE_DECISIONS


def score_single_decision(domain, weighted_attributes, rl_memory=None, session_history=None, profile=None):
    """
    Compute an 11-term fairness breakdown for a single decision.
    
    Returns:
        {
            'score': float (0-1),
            'grade': str ('A' through 'F'),
            'breakdown': {term: value, ...},
            'total_reward': float (sum of all terms),
            'interpretation': str
        }
    """
    domain_lower = domain.lower()
    breakdown = {}
    
    protected = PROTECTED_ATTRIBUTES.get(domain_lower, [])
    proxies = PROXY_MAP.get(domain_lower, [])
    merits = MERIT_ATTRIBUTES.get(domain_lower, [])
    redundants = REDUNDANT_ATTRIBUTES.get(domain_lower, [])
    
    # Extract attribute names
    attr_names = []
    for a in weighted_attributes:
        if isinstance(a, dict):
            attr_names.append(a.get("attribute", ""))
        elif isinstance(a, str):
            attr_names.append(a)
    
    # ── POSITIVE TERMS ──
    
    # 1. Fair attribute weight: bonus for each merit-based attribute used
    merit_count = sum(1 for a in attr_names if _ci_match(a, merits))
    fair_weight = min(merit_count * 0.08, 0.25)
    if fair_weight > 0:
        breakdown["fair_attribute_weight"] = round(fair_weight, 3)
    
    # 2. Protected attribute ignored: bonus if NO protected attributes used
    protected_used = [a for a in attr_names if _ci_match(a, protected)]
    if len(protected_used) == 0 and len(attr_names) > 0:
        breakdown["protected_attr_ignored"] = 0.15
    
    # 3. RL alignment: bonus if decision aligns with rewarded patterns
    rl_bonus = 0.0
    if rl_memory:
        pos_list = rl_memory.get(domain_lower, {}).get("positive", [])
        neg_list = rl_memory.get(domain_lower, {}).get("negative", [])
        for a in attr_names:
            if a in pos_list:
                rl_bonus += 0.05
            if a in neg_list:
                rl_bonus -= 0.05
    if rl_bonus != 0:
        breakdown["rl_alignment"] = round(max(-0.2, min(0.2, rl_bonus)), 3)
    
    # 4. Transparency: bonus if attributes have reasoning
    has_reasoning = sum(1 for a in weighted_attributes
                        if isinstance(a, dict) and a.get("reasoning"))
    if has_reasoning > 0:
        breakdown["transparency_score"] = round(min(has_reasoning * 0.04, 0.12), 3)
    
    # 5. Diverse outcome: bonus if session history shows diversity
    if session_history and len(session_history) >= 3:
        positive_decisions = sum(1 for h in session_history
                                 if _is_positive_decision(h.get("decision", "")))
        total = len(session_history)
        ratio = positive_decisions / total if total > 0 else 0.5
        if 0.3 <= ratio <= 0.7:
            breakdown["diverse_outcome"] = 0.10
    
    # ── NEGATIVE TERMS ──
    
    # 6. Protected attribute weighted: penalty per protected attribute used
    if protected_used:
        penalty = min(len(protected_used) * -0.15, -0.30)
        breakdown["protected_attr_weighted"] = round(penalty, 3)
    
    # 7. Proxy leak: penalty for proxy attributes
    proxy_used = [a for a in attr_names if _ci_match(a, proxies)]
    if proxy_used:
        penalty = min(len(proxy_used) * -0.10, -0.20)
        breakdown["proxy_leak"] = round(penalty, 3)
    
    # 8. Redundant attribute used
    redundant_used = [a for a in attr_names if _ci_match(a, redundants)]
    if redundant_used:
        penalty = min(len(redundant_used) * -0.08, -0.16)
        breakdown["redundant_attr_used"] = round(penalty, 3)
    
    # 9. Outcome skew: penalty if session outcomes skew to one demographic
    if session_history and len(session_history) >= 5:
        positive_decisions = sum(1 for h in session_history
                                 if _is_positive_decision(h.get("decision", "")))
        total = len(session_history)
        ratio = positive_decisions / total if total > 0 else 0.5
        if ratio > 0.85 or ratio < 0.15:
            breakdown["outcome_skew"] = -0.12
    
    # 10. Inaction streak: penalty if user hasn't given feedback
    if session_history and len(session_history) >= 4:
        recent = session_history[-4:]
        no_feedback = sum(1 for h in recent if not h.get("feedback_given", False))
        if no_feedback >= 3:
            breakdown["inaction_streak"] = -0.08
    
    # 11. Unknown attribute risk: penalty when decision uses unclassified attributes
    #     These are attributes the model was NOT trained on / not in any known list.
    #     We classify the profile (if available) or check attr_names against known lists.
    unknown_attrs = []
    if profile:
        # Use the full classifier to detect unknowns
        classifications = classify_attributes(domain_lower, profile, use_gemini=False)
        unknown_attrs = [a for a in attr_names if classifications.get(a) == "UNKNOWN"]
    else:
        # Fallback: check if attribute is in any known list
        all_known = set(
            PROTECTED_ATTRIBUTES.get(domain_lower, []) +
            PROXY_MAP.get(domain_lower, []) +
            MERIT_ATTRIBUTES.get(domain_lower, []) +
            REDUNDANT_ATTRIBUTES.get(domain_lower, [])
        )
        unknown_attrs = [a for a in attr_names if a not in all_known]
    
    if unknown_attrs:
        penalty = max(len(unknown_attrs) * -0.06, -0.18)
        breakdown["unknown_attr_risk"] = round(penalty, 3)
    
    # ── COMPUTE FINAL SCORE ──
    total_reward = sum(breakdown.values())
    raw_score = 0.5 + total_reward  # Center at 0.5
    score = max(0.0, min(1.0, raw_score))
    
    # Letter grade
    if score >= 0.85:
        grade = "A"
    elif score >= 0.70:
        grade = "B"
    elif score >= 0.55:
        grade = "C"
    elif score >= 0.40:
        grade = "D"
    else:
        grade = "F"
    
    # Interpretation
    if unknown_attrs:
        unknown_warning = f" ⚠️ Unverified attributes detected: {', '.join(unknown_attrs)}."
    else:
        unknown_warning = ""
    
    if score >= 0.80:
        interpretation = "This decision shows strong fairness — merit-based attributes drive the outcome." + unknown_warning
    elif score >= 0.60:
        interpretation = "Moderate fairness. Some bias signals detected but within acceptable range." + unknown_warning
    elif score >= 0.40:
        interpretation = "Fairness concerns detected. Protected or proxy attributes may be influencing the decision." + unknown_warning
    else:
        interpretation = "Significant bias detected. Protected attributes are heavily influencing this decision." + unknown_warning
    
    return {
        "score": round(score, 3),
        "grade": grade,
        "breakdown": breakdown,
        "total_reward": round(total_reward, 3),
        "interpretation": interpretation,
        "positive_terms": {k: v for k, v in breakdown.items() if v > 0},
        "negative_terms": {k: v for k, v in breakdown.items() if v < 0},
        "unknown_attributes": unknown_attrs,
    }


def grade_session(session_history, domain, rl_memory=None):
    """
    Compute an overall session fairness grade (0-1).
    Inspired by Financial Triage's grade_episode().
    
    Weighted criteria:
      30% Disparate Impact closeness to 1.0
      20% Outcome Balance  
      15% Attribute Hygiene
      15% Feedback Engagement
      10% RL Alignment
      10% Proxy Resistance
    """
    if not session_history or len(session_history) < 2:
        return {
            "grade": "N/A",
            "score": 0.0,
            "message": "Need at least 2 decisions to grade the session",
            "criteria": {},
        }
    
    domain_lower = domain.lower()
    protected = PROTECTED_ATTRIBUTES.get(domain_lower, [])
    proxies = PROXY_MAP.get(domain_lower, [])
    criteria = {}
    
    # 1. Outcome Balance (20%)
    positive = sum(1 for h in session_history
                   if _is_positive_decision(h.get("decision", "")))
    total = len(session_history)
    ratio = positive / total if total > 0 else 0.5
    balance_score = 1.0 - abs(ratio - 0.5) * 2  # Perfect at 50/50
    criteria["outcome_balance"] = round(balance_score, 3)
    
    # 2. Attribute Hygiene (15%)
    clean_count = 0
    for h in session_history:
        attrs = h.get("weighted_attributes", [])
        attr_names = [a.get("attribute", "") if isinstance(a, dict) else a for a in attrs]
        if not any(_ci_match(a, protected) for a in attr_names):
            clean_count += 1
    hygiene = clean_count / total if total > 0 else 1.0
    criteria["attribute_hygiene"] = round(hygiene, 3)
    
    # 3. Feedback Engagement (15%)
    feedback_count = sum(1 for h in session_history if h.get("feedback_given", False))
    engagement = feedback_count / total if total > 0 else 0.0
    criteria["feedback_engagement"] = round(engagement, 3)
    
    # 4. Proxy Resistance (10%)
    proxy_clean = 0
    for h in session_history:
        attrs = h.get("weighted_attributes", [])
        attr_names = [a.get("attribute", "") if isinstance(a, dict) else a for a in attrs]
        if not any(_ci_match(a, proxies) for a in attr_names):
            proxy_clean += 1
    proxy_resist = proxy_clean / total if total > 0 else 1.0
    criteria["proxy_resistance"] = round(proxy_resist, 3)
    
    # 5. RL Alignment (10%)
    rl_score = 0.5  # Default neutral
    if rl_memory:
        neg_list = rl_memory.get(domain_lower, {}).get("negative", [])
        if neg_list:
            violations = 0
            for h in session_history:
                attrs = h.get("weighted_attributes", [])
                attr_names = [a.get("attribute", "") if isinstance(a, dict) else a for a in attrs]
                if any(a in neg_list for a in attr_names):
                    violations += 1
            rl_score = 1.0 - (violations / total)
    criteria["rl_alignment"] = round(rl_score, 3)
    
    # 6. Disparate Impact estimate (30%) — simplified
    # Without actual AIF360 batch run, estimate from outcome ratios
    di_score = balance_score  # Use outcome balance as proxy
    criteria["disparate_impact_est"] = round(di_score, 3)
    
    # Weighted sum
    final = (
        0.30 * di_score +
        0.20 * balance_score +
        0.15 * hygiene +
        0.15 * engagement +
        0.10 * rl_score +
        0.10 * proxy_resist
    )
    final = round(max(0.0, min(1.0, final)), 3)
    
    if final >= 0.85:
        grade = "A"
    elif final >= 0.70:
        grade = "B"
    elif final >= 0.55:
        grade = "C"
    elif final >= 0.40:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "grade": grade,
        "score": final,
        "message": f"Session fairness grade: {grade} ({final:.1%})",
        "criteria": criteria,
        "total_decisions": total,
    }
