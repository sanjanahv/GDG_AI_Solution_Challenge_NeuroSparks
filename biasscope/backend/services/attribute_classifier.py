# backend/services/attribute_classifier.py
# ============================================================
# Attribute Classifier: NORMAL | REDUNDANT | AMBIGUOUS | PROTECTED | UNKNOWN
# ============================================================
# Classifies each attribute in a profile to help users understand
# which attributes are safe vs risky for AI decision-making.
#
# NORMAL:    Directly relevant, no proxy for protected attributes
# REDUNDANT: Duplicates another attribute, adds no new info
# AMBIGUOUS: Correlates with protected attributes (proxy bias)
# PROTECTED: Directly protected attribute (gender, race, etc.)
# UNKNOWN:   Not in any known list — flagged for review
#
# NEW: Dynamic proxy detection via Gemini for unknown attributes
# ============================================================

import os
import json
import logging

logger = logging.getLogger(__name__)

# ─── Gemini dynamic proxy detection ──────────────────────────────

try:
    import google.generativeai as genai
    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if _GEMINI_API_KEY:
        genai.configure(api_key=_GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        logger.warning("[WARN] GEMINI_API_KEY not set — dynamic proxy detection disabled")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("[WARN] google-generativeai not installed — dynamic proxy detection disabled")

# ─── Cache for Gemini proxy detection results ────────────────────
# Prevents repeated API calls for the same attribute+domain pair.
# Key: (domain, attribute_lowercase), Value: classification string
_dynamic_classification_cache = {}

# ─── Learned proxy attributes from RL penalization feedback ──────
# When a user penalizes an UNKNOWN attribute, it gets added here
# so future classifications don't need a Gemini call.
_learned_proxies = {
    "job": [],
    "loan": [],
    "college": [],
}

_LEARNED_PROXIES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "learned_proxies.json"
)


def _load_learned_proxies():
    """Load learned proxy attributes from disk."""
    global _learned_proxies
    try:
        if os.path.exists(_LEARNED_PROXIES_FILE):
            with open(_LEARNED_PROXIES_FILE, "r") as f:
                _learned_proxies = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass


def _save_learned_proxies():
    """Persist learned proxy attributes to disk."""
    os.makedirs(os.path.dirname(_LEARNED_PROXIES_FILE), exist_ok=True)
    with open(_LEARNED_PROXIES_FILE, "w") as f:
        json.dump(_learned_proxies, f, indent=2)


# Load on import
_load_learned_proxies()


# ─── Static classification lists ─────────────────────────────────

# Known ambiguous attributes per domain (proxy bias risks)
AMBIGUOUS_MAP = {
    "job": [
        "zip_code", "Zip Code", "ZipCode",
        "university_name", "University", "University Name",
        "graduation_year", "Graduation Year",
        "hobbies", "Hobbies",
        "photo", "Photo",
        "Location", "location",
        "Marital Status", "marital_status",
        "Name", "name",
    ],
    "loan": [
        "zip_code", "Zip Code", "ZipCode",
        "neighborhood", "Neighborhood",
        "employer_name", "Employer", "Employer Name",
        "spending_category", "Spending Category",
        "Location", "location",
        "Name", "name",
    ],
    "college": [
        "high_school_name", "High School", "High School Name",
        "extracurriculars", "Extracurriculars",
        "zip_code", "Zip Code", "ZipCode",
        "parent_occupation", "Parent Occupation",
        "Legacy Status", "legacy_status",
        "Location", "location",
        "Name", "name",
    ],
}

# Known redundant pairs (if both present, second one is redundant)
REDUNDANT_PAIRS = {
    "job": [
        ("years_experience", "number_of_jobs"),
        ("Years of Experience", "Number of Jobs"),
        ("Experience Level", "Years of Experience"),
    ],
    "loan": [
        ("monthly_income", "annual_income"),
        ("Monthly Income", "Annual Income"),
    ],
    "college": [
        ("gpa", "class_rank"),
        ("GPA", "Class Rank"),
    ],
}

# Protected attributes that should ALWAYS be flagged
PROTECTED_ATTRS = [
    "Gender", "gender", "Sex", "sex",
    "Race", "race", "Ethnicity", "ethnicity",
    "Religion", "religion",
    "Sexual Orientation", "sexual_orientation",
    "Disability", "disability",
    "Nationality", "nationality",
    "Age", "age",
]

# Known merit-based attributes that are definitely NORMAL
# (prevents false-positive UNKNOWN flags on common safe attributes)
KNOWN_MERIT_ATTRS = {
    "job": [
        "Experience", "Skills", "Education Level", "Projects", "Certifications",
        "Years of Experience", "Technical Skills", "Leadership", "Performance Rating",
        "References", "Technical Score", "Projects Completed", "Years Experience",
        "Applied Role", "applied_role", "Previous Industry", "previous_industry",
        "Highest Education", "highest_education", "skills",
    ],
    "loan": [
        "Credit Score", "Annual Income", "Employment Status", "Debt-to-Income",
        "Loan Amount", "Payment History", "Employment Length", "Savings Balance",
        "Assets", "Debt to Income Ratio", "credit_score", "annual_income",
        "employment_status", "loan_amount_requested", "loan_purpose",
        "Loan Purpose", "existing_debt", "Existing Debt",
    ],
    "college": [
        "GPA", "SAT Score", "ACT Score", "Extracurriculars", "Awards",
        "Research", "Volunteer Hours", "AP Classes", "Essay Score",
        "Recommendation Strength", "Leadership Roles",
        "gpa", "sat_score", "applied_major", "Applied Major",
    ],
}


# ─── Gemini dynamic proxy detection ──────────────────────────────

def _detect_proxy_via_gemini(domain: str, attribute: str) -> str:
    """
    Use Gemini to dynamically classify an unknown attribute.

    Returns one of: "AMBIGUOUS", "NORMAL"
    Falls back to "UNKNOWN" if Gemini is unavailable or errors.
    """
    if not GEMINI_AVAILABLE:
        return "UNKNOWN"

    cache_key = (domain.lower(), attribute.lower())
    if cache_key in _dynamic_classification_cache:
        return _dynamic_classification_cache[cache_key]

    prompt = f"""You are a fairness auditor analyzing attributes used in AI decision-making.

Domain: {domain}
Attribute: "{attribute}"

Could this attribute be a PROXY for any protected attribute (Gender, Race, Age, Religion, Disability, Socioeconomic Status, Nationality)?

A proxy attribute is one that correlates with or can predict a protected attribute, even if it seems neutral. For example:
- "Zip Code" → proxy for Race (due to housing segregation)
- "University Name" → proxy for Wealth/Socioeconomic Status
- "First Name" → proxy for Gender or Ethnicity

Respond with ONLY a JSON object, no markdown:
{{"classification": "AMBIGUOUS" or "NORMAL", "proxy_for": "which protected attribute it proxies for, or null", "confidence": 0.0 to 1.0, "reasoning": "one sentence explanation"}}"""

    try:
        response = _gemini_model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(text)

        classification = result.get("classification", "UNKNOWN")
        confidence = result.get("confidence", 0.5)

        # Only trust high-confidence Gemini classifications
        if confidence >= 0.7 and classification in ("AMBIGUOUS", "NORMAL"):
            _dynamic_classification_cache[cache_key] = classification
            logger.info(
                f"[Gemini Proxy Detection] '{attribute}' in {domain} → "
                f"{classification} (conf={confidence}, proxy_for={result.get('proxy_for')})"
            )
            return classification
        else:
            # Low confidence → flag as UNKNOWN for human review
            _dynamic_classification_cache[cache_key] = "UNKNOWN"
            return "UNKNOWN"

    except Exception as e:
        logger.warning(f"[Gemini Proxy Detection] Failed for '{attribute}': {e}")
        _dynamic_classification_cache[cache_key] = "UNKNOWN"
        return "UNKNOWN"


# ─── Core classification function ────────────────────────────────

def classify_attributes(domain: str, attributes: dict, use_gemini: bool = True) -> dict:
    """
    Classify each attribute as NORMAL, REDUNDANT, AMBIGUOUS, PROTECTED, or UNKNOWN.

    Args:
        domain: 'job', 'loan', or 'college'
        attributes: dict of attribute name -> value
        use_gemini: if True, uses Gemini to dynamically classify unknown attributes

    Returns:
        dict of attribute name -> classification string
    """
    d = domain.lower()
    ambiguous_set = set(AMBIGUOUS_MAP.get(d, []))
    merit_set = set(KNOWN_MERIT_ATTRS.get(d, []))
    learned_set = set(_learned_proxies.get(d, []))
    result = {}

    for attr in attributes:
        if attr in PROTECTED_ATTRS:
            result[attr] = "PROTECTED"
        elif attr in ambiguous_set:
            result[attr] = "AMBIGUOUS"
        elif attr in learned_set:
            # Previously penalized unknown attribute → now known AMBIGUOUS
            result[attr] = "AMBIGUOUS"
        elif attr in merit_set:
            result[attr] = "NORMAL"
        else:
            # ── Unknown attribute: not in any static list ──
            if use_gemini:
                result[attr] = _detect_proxy_via_gemini(d, attr)
            else:
                result[attr] = "UNKNOWN"

    # Check for redundancy
    for (a, b) in REDUNDANT_PAIRS.get(d, []):
        if a in result and b in result:
            result[b] = "REDUNDANT"

    return result


def get_classification_summary(domain: str, attributes: dict, use_gemini: bool = True) -> dict:
    """
    Get a summary of attribute classifications with explanations.
    """
    classifications = classify_attributes(domain, attributes, use_gemini)

    summary = {
        "classifications": classifications,
        "counts": {
            "normal": sum(1 for v in classifications.values() if v == "NORMAL"),
            "ambiguous": sum(1 for v in classifications.values() if v == "AMBIGUOUS"),
            "redundant": sum(1 for v in classifications.values() if v == "REDUNDANT"),
            "protected": sum(1 for v in classifications.values() if v == "PROTECTED"),
            "unknown": sum(1 for v in classifications.values() if v == "UNKNOWN"),
        },
        "warnings": [],
    }

    # Generate warnings
    for attr, cls in classifications.items():
        if cls == "AMBIGUOUS":
            summary["warnings"].append(
                f"'{attr}' may be a proxy for protected attributes (e.g., zip_code → race). "
                f"Consider penalizing if the AI weights it heavily."
            )
        elif cls == "PROTECTED":
            summary["warnings"].append(
                f"'{attr}' is a protected attribute. AI should NOT use this for decision-making."
            )
        elif cls == "REDUNDANT":
            summary["warnings"].append(
                f"'{attr}' appears redundant — it duplicates information from another attribute."
            )
        elif cls == "UNKNOWN":
            summary["warnings"].append(
                f"⚠️ '{attr}' is a NEW attribute not seen before. "
                f"It has not been verified as safe. Review it and use Reward/Penalize to classify it."
            )

    return summary


# ─── Learning from RL feedback ────────────────────────────────────

def learn_proxy_from_feedback(domain: str, attribute: str, action: str):
    """
    Called when a user penalizes or rewards an UNKNOWN attribute.
    Updates the learned proxies list so future classifications
    don't require a Gemini call.

    Args:
        domain: 'job', 'loan', or 'college'
        attribute: the attribute name
        action: 'penalize' or 'reward'
    """
    d = domain.lower()
    if d not in _learned_proxies:
        _learned_proxies[d] = []

    if action == "penalize":
        # User identified this as biased → add to learned proxies
        if attribute not in _learned_proxies[d]:
            _learned_proxies[d].append(attribute)
            logger.info(f"[Learned Proxy] Added '{attribute}' to {d} proxy list via user penalization")
        # Also update the dynamic cache
        cache_key = (d, attribute.lower())
        _dynamic_classification_cache[cache_key] = "AMBIGUOUS"

    elif action == "reward":
        # User confirmed this is safe → remove from learned proxies if present
        _learned_proxies[d] = [a for a in _learned_proxies[d] if a != attribute]
        # Update cache
        cache_key = (d, attribute.lower())
        _dynamic_classification_cache[cache_key] = "NORMAL"

    _save_learned_proxies()


def get_learned_proxies():
    """Return all learned proxy attributes for display/debugging."""
    return _learned_proxies


def clear_dynamic_cache():
    """Clear the Gemini classification cache (useful for testing)."""
    _dynamic_classification_cache.clear()
