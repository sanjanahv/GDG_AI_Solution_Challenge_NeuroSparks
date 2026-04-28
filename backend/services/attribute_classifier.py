"""
Attribute Classifier — Normal / Redundant / Ambiguous / Protected

Classifies profile attributes into risk categories:
  - NORMAL: Directly relevant, no proxy for protected attributes (Skills, GPA)
  - AMBIGUOUS: Correlates with protected attributes — proxy bias (zip_code → race)
  - REDUNDANT: Duplicates another attribute, adds no info (annual + monthly income)
  - PROTECTED: Directly a protected attribute (Gender, Race, Age)

This is the academic differentiator of BiasScope.
"""

# ─── Protected Attributes (directly sensitive) ───────────────────

PROTECTED_ATTRIBUTES = {
    "gender", "sex", "race", "ethnicity", "religion",
    "age", "disability", "marital_status", "sexual_orientation",
    "national_origin", "nationality", "pregnancy_status"
}

# ─── Known ambiguous attributes per domain (proxy bias risks) ────

AMBIGUOUS_MAP = {
    "job": [
        "zip_code", "zipcode", "zip", "postal_code",
        "university_name", "university", "college_name", "alma_mater",
        "graduation_year", "grad_year",
        "hobbies", "interests",
        "photo", "profile_photo",
        "name", "first_name", "last_name",
        "address", "city", "location"
    ],
    "loan": [
        "zip_code", "zipcode", "zip", "postal_code",
        "neighborhood", "area", "locality",
        "employer_name", "company_name",
        "spending_category", "spending_pattern",
        "name", "first_name", "last_name",
        "address", "city", "location"
    ],
    "college": [
        "high_school_name", "high_school", "school_name",
        "extracurriculars", "extracurricular_activities", "activities",
        "zip_code", "zipcode", "zip", "postal_code",
        "parent_occupation", "parent_job", "parents_occupation",
        "parent_income", "family_income",
        "name", "first_name", "last_name",
        "address", "city", "location"
    ]
}

# ─── Known redundant pairs (if both present, second is redundant) ─

REDUNDANT_PAIRS = {
    "job": [
        ("years_experience", "number_of_jobs"),
        ("education_level", "degree"),
        ("skills", "certifications")
    ],
    "loan": [
        ("monthly_income", "annual_income"),
        ("credit_score", "credit_rating"),
        ("loan_amount", "loan_amount_requested")
    ],
    "college": [
        ("gpa", "class_rank"),
        ("sat_score", "act_score"),
        ("gpa", "academic_percentile")
    ]
}

# ─── Proxy Bias Explanations ─────────────────────────────────────

PROXY_EXPLANATIONS = {
    "zip_code": "Correlates with race due to historical housing segregation patterns.",
    "zipcode": "Correlates with race due to historical housing segregation patterns.",
    "zip": "Correlates with race due to historical housing segregation patterns.",
    "postal_code": "Correlates with race due to historical housing segregation patterns.",
    "university_name": "Correlates with socioeconomic status and wealth/privilege.",
    "university": "Correlates with socioeconomic status and wealth/privilege.",
    "college_name": "Correlates with socioeconomic status and wealth/privilege.",
    "alma_mater": "Correlates with socioeconomic status and wealth/privilege.",
    "graduation_year": "Can serve as a proxy for age discrimination.",
    "grad_year": "Can serve as a proxy for age discrimination.",
    "high_school_name": "Correlates with neighborhood wealth and racial demographics.",
    "high_school": "Correlates with neighborhood wealth and racial demographics.",
    "school_name": "Correlates with neighborhood wealth and racial demographics.",
    "parent_occupation": "Proxy for family socioeconomic status.",
    "parent_job": "Proxy for family socioeconomic status.",
    "parent_income": "Direct proxy for socioeconomic privilege.",
    "family_income": "Direct proxy for socioeconomic privilege.",
    "neighborhood": "Correlates with race and income level.",
    "area": "Correlates with race and income level.",
    "locality": "Correlates with race and income level.",
    "employer_name": "May correlate with socioeconomic networks and privilege.",
    "company_name": "May correlate with socioeconomic networks and privilege.",
    "extracurriculars": "Access to activities correlates with family wealth.",
    "hobbies": "Certain hobbies correlate with socioeconomic background.",
    "interests": "Certain interests correlate with socioeconomic background.",
    "name": "Names can reveal ethnicity, gender, or national origin.",
    "first_name": "First names can reveal gender and ethnicity.",
    "last_name": "Last names can reveal ethnicity and national origin.",
    "address": "Address correlates with race and socioeconomic status.",
    "city": "City of residence correlates with demographics.",
    "location": "Location correlates with demographics and opportunity access.",
    "photo": "Photos reveal race, gender, age, and other protected attributes.",
    "profile_photo": "Photos reveal race, gender, age, and other protected attributes.",
    "spending_category": "Spending patterns can correlate with cultural background.",
    "spending_pattern": "Spending patterns can correlate with cultural background.",
}


def _normalize_key(attr_name: str) -> str:
    """Normalize attribute name for lookup: lowercase, replace spaces with underscores."""
    return attr_name.lower().strip().replace(" ", "_").replace("-", "_")


def classify_attributes(domain: str, attributes: dict) -> dict:
    """
    Classify each attribute in a profile as NORMAL, AMBIGUOUS, REDUNDANT, or PROTECTED.

    Args:
        domain: "job", "loan", or "college"
        attributes: dict of attribute_name → value

    Returns:
        dict of attribute_name → classification string
    """
    d = domain.lower()
    ambiguous_set = set(_normalize_key(a) for a in AMBIGUOUS_MAP.get(d, []))
    result = {}

    for attr in attributes:
        normalized = _normalize_key(attr)

        # Check protected first (highest priority)
        if normalized in PROTECTED_ATTRIBUTES:
            result[attr] = "PROTECTED"
        # Check ambiguous (proxy bias)
        elif normalized in ambiguous_set:
            result[attr] = "AMBIGUOUS"
        else:
            result[attr] = "NORMAL"

    # Check redundancy (if both attrs in a pair exist, second is redundant)
    for (a, b) in REDUNDANT_PAIRS.get(d, []):
        # Check if both exist (using normalized matching)
        a_found = any(_normalize_key(attr) == a for attr in attributes)
        b_found = any(_normalize_key(attr) == a for attr in attributes) and \
                  any(_normalize_key(attr) == b for attr in attributes)
        if b_found:
            # Find the actual key that matches b and mark it redundant
            for attr in attributes:
                if _normalize_key(attr) == b:
                    result[attr] = "REDUNDANT"

    return result


def get_proxy_explanation(attribute: str) -> str:
    """Get the proxy bias explanation for an attribute, or empty string if not ambiguous."""
    normalized = _normalize_key(attribute)
    return PROXY_EXPLANATIONS.get(normalized, "")


def get_protected_attrs_used(weighted_attributes: list[dict]) -> list[str]:
    """
    From a list of weighted attributes, return those that are protected.
    Used by the fairness scorer to detect when protected attributes drive decisions.
    """
    protected = []
    for wa in weighted_attributes:
        attr_name = wa.get("attribute", "")
        if _normalize_key(attr_name) in PROTECTED_ATTRIBUTES:
            protected.append(attr_name)
    return protected


def get_proxy_attrs_used(domain: str, weighted_attributes: list[dict]) -> list[str]:
    """
    From a list of weighted attributes, return those that are ambiguous (proxy).
    """
    d = domain.lower()
    ambiguous_set = set(_normalize_key(a) for a in AMBIGUOUS_MAP.get(d, []))
    proxies = []
    for wa in weighted_attributes:
        attr_name = wa.get("attribute", "")
        if _normalize_key(attr_name) in ambiguous_set:
            proxies.append(attr_name)
    return proxies
