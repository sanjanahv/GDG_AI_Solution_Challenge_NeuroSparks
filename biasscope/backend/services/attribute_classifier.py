# backend/services/attribute_classifier.py
# ============================================================
# Attribute Classifier: NORMAL | REDUNDANT | AMBIGUOUS
# ============================================================
# Classifies each attribute in a profile to help users understand
# which attributes are safe vs risky for AI decision-making.
#
# NORMAL:    Directly relevant, no proxy for protected attributes
# REDUNDANT: Duplicates another attribute, adds no new info
# AMBIGUOUS: Correlates with protected attributes (proxy bias)
# ============================================================

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


def classify_attributes(domain: str, attributes: dict) -> dict:
    """
    Classify each attribute as NORMAL, REDUNDANT, AMBIGUOUS, or PROTECTED.

    Args:
        domain: 'job', 'loan', or 'college'
        attributes: dict of attribute name -> value

    Returns:
        dict of attribute name -> classification string
    """
    d = domain.lower()
    ambiguous_set = set(AMBIGUOUS_MAP.get(d, []))
    result = {}

    for attr in attributes:
        if attr in PROTECTED_ATTRS:
            result[attr] = "PROTECTED"
        elif attr in ambiguous_set:
            result[attr] = "AMBIGUOUS"
        else:
            result[attr] = "NORMAL"

    # Check for redundancy
    for (a, b) in REDUNDANT_PAIRS.get(d, []):
        if a in result and b in result:
            result[b] = "REDUNDANT"

    return result


def get_classification_summary(domain: str, attributes: dict) -> dict:
    """
    Get a summary of attribute classifications with explanations.
    """
    classifications = classify_attributes(domain, attributes)

    summary = {
        "classifications": classifications,
        "counts": {
            "normal": sum(1 for v in classifications.values() if v == "NORMAL"),
            "ambiguous": sum(1 for v in classifications.values() if v == "AMBIGUOUS"),
            "redundant": sum(1 for v in classifications.values() if v == "REDUNDANT"),
            "protected": sum(1 for v in classifications.values() if v == "PROTECTED"),
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

    return summary
