# backend/services/aif360_service.py
# ============================================================
# AIF360 Post-Processing Bias Detection & Correction Service
# ============================================================
# This is the CORE of BiasScope's bias detection.
# It uses IBM AIF360's POST-PROCESSING algorithms to:
#   1. Measure bias in a batch of AI decisions
#   2. Correct biased decisions WITHOUT retraining the model
#
# Post-Processing Algorithms Used:
#   - CalibratedEqOddsPostprocessing (default)
#   - EqOddsPostprocessing (strict)
#   - RejectOptionClassification (borderline flip)
# ============================================================

import pandas as pd
import numpy as np

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.postprocessing import (
        CalibratedEqOddsPostprocessing,
        EqOddsPostprocessing,
        RejectOptionClassification
    )
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("[WARN] aif360 not installed — using manual bias metrics fallback")


# --- Decision label mappings per domain ---
DECISION_MAP = {
    "Hire": 1, "Reject": 0,
    "Approve": 1, "Deny": 0,
    "Admit": 1, "Waitlist": 0,
    "Accept": 1, "Decline": 0,
    # Lowercase variants
    "hire": 1, "reject": 0,
    "approve": 1, "deny": 0,
    "admit": 1, "waitlist": 0,
}

# --- Protected attributes per domain ---
PROTECTED_ATTRIBUTES = {
    "job": ["Gender", "Age"],
    "loan": ["Gender", "Race"],
    "college": ["Gender", "Parent Occupation"],
}

# --- Encoding maps for protected attributes ---
GENDER_MAP = {
    "Male": 1, "Female": 0, "male": 1, "female": 0,
    "M": 1, "F": 0, "Non-binary": 0, "Other": 0,
    "Non-Binary": 0, "non-binary": 0
}

RACE_MAP = {
    "White": 1, "Caucasian": 1, "Asian": 0, "Black": 0,
    "African American": 0, "Hispanic": 0, "Latino": 0,
    "Native American": 0, "Other": 0, "Mixed": 0,
    "white": 1, "caucasian": 1, "asian": 0, "black": 0,
}


def _encode_protected_attribute(value, attr_name):
    """Convert a protected attribute value to binary (0/1)."""
    attr_lower = attr_name.lower()
    if "gender" in attr_lower or "sex" in attr_lower:
        return GENDER_MAP.get(str(value), 0)
    if "race" in attr_lower or "ethnicity" in attr_lower:
        return RACE_MAP.get(str(value), 0)
    if "age" in attr_lower:
        try:
            age = int(value)
            return 1 if age < 40 else 0  # Young=1 (privileged), Old=0
        except (ValueError, TypeError):
            return 0
    # Default: try to interpret as binary
    if isinstance(value, (int, float)):
        return 1 if value > 0 else 0
    return 0


def _build_dataset(records, protected_attr, label_col="decision"):
    """
    Build an AIF360 BinaryLabelDataset from a list of decision records.

    Each record should have:
      - profile attributes (including the protected attribute)
      - a 'decision' field (e.g., "Hire", "Reject")
    """
    rows = []
    for r in records:
        attrs = r.get("attributes", r.get("profile", {}))
        decision_raw = r.get("decision", r.get("label", 0))

        # Encode the decision
        if isinstance(decision_raw, str):
            decision_val = DECISION_MAP.get(decision_raw, 0)
        else:
            decision_val = int(decision_raw)

        # Get protected attribute value
        prot_val = attrs.get(protected_attr, None)
        if prot_val is None:
            # Try case-insensitive lookup
            for k, v in attrs.items():
                if k.lower() == protected_attr.lower():
                    prot_val = v
                    break
        if prot_val is None:
            continue  # Skip records missing the protected attribute

        encoded_prot = _encode_protected_attribute(prot_val, protected_attr)

        rows.append({
            protected_attr: encoded_prot,
            label_col: decision_val,
        })

    if len(rows) < 2:
        return None, None, None

    df = pd.DataFrame(rows)

    # Need at least one record in each group
    if df[protected_attr].nunique() < 2:
        return None, None, None

    dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_col],
        protected_attribute_names=[protected_attr],
        favorable_label=1,
        unfavorable_label=0
    )

    privileged = [{protected_attr: 1}]
    unprivileged = [{protected_attr: 0}]

    return dataset, privileged, unprivileged


def compute_bias_metrics(records, protected_attr="Gender", label_col="decision"):
    """
    Compute bias metrics for a batch of decision records.

    Returns:
        dict with disparate_impact, statistical_parity_diff, bias_detected, etc.
    """
    if not AIF360_AVAILABLE:
        # Manual DI calculation fallback
        priv_pos = priv_tot = unpriv_pos = unpriv_tot = 0
        for r in records:
            attrs = r.get("attributes", r.get("profile", r))
            pv = None
            for k, v in attrs.items():
                if k.lower() == protected_attr.lower():
                    pv = v; break
            if pv is None: continue
            enc = _encode_protected_attribute(pv, protected_attr)
            dec_raw = r.get("decision", "")
            dec = DECISION_MAP.get(str(dec_raw), 0) if isinstance(dec_raw, str) else int(dec_raw)
            if enc == 1:
                priv_tot += 1; priv_pos += dec
            else:
                unpriv_tot += 1; unpriv_pos += dec
        pr = priv_pos / max(priv_tot, 1)
        ur = unpriv_pos / max(unpriv_tot, 1)
        di = ur / max(pr, 0.001)
        spd = ur - pr
        bd = di < 0.8 or di > 1.25
        return {
            "disparate_impact": round(di, 4), "statistical_parity_diff": round(spd, 4),
            "bias_detected": bd, "severity": "high" if (di < 0.5 or di > 2.0) else ("moderate" if bd else "none"),
            "num_records": len(records), "protected_attribute": protected_attr, "method": "manual",
            "threshold_info": {"fair_range": "0.80-1.25", "current_value": round(di, 4),
                "interpretation": "FAIR" if not bd else f"BIASED — DI {round(di,4)}"}
        }

    dataset, privileged, unprivileged = _build_dataset(records, protected_attr, label_col)

    if dataset is None:
        return {
            "disparate_impact": None,
            "statistical_parity_diff": None,
            "bias_detected": False,
            "error": "Not enough data — need at least 2 records with both privileged and unprivileged groups.",
            "num_records": len(records),
            "protected_attribute": protected_attr,
        }

    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )

    di = metric.disparate_impact()
    spd = metric.statistical_parity_difference()

    # Handle edge cases (inf, nan)
    if np.isinf(di) or np.isnan(di):
        di = 0.0
    if np.isinf(spd) or np.isnan(spd):
        spd = 0.0

    # 80% rule: disparate impact < 0.8 or > 1.25 = bias detected
    bias_detected = di < 0.8 or di > 1.25

    # Determine bias direction
    if di < 0.8:
        bias_direction = "negative"  # Unprivileged group is disadvantaged
        severity = "high" if di < 0.5 else "moderate"
    elif di > 1.25:
        bias_direction = "positive"  # Unprivileged group is over-favored
        severity = "high" if di > 2.0 else "moderate"
    else:
        bias_direction = "none"
        severity = "none"

    return {
        "disparate_impact": round(float(di), 4),
        "statistical_parity_diff": round(float(spd), 4),
        "bias_detected": bias_detected,
        "bias_direction": bias_direction,
        "severity": severity,
        "num_records": len(records),
        "protected_attribute": protected_attr,
        "threshold_info": {
            "fair_range": "0.80 - 1.25",
            "current_value": round(float(di), 4),
            "interpretation": (
                "FAIR — No significant bias detected"
                if not bias_detected else
                f"BIASED — Disparate impact {round(float(di), 4)} is outside the fair range [0.80, 1.25]"
            )
        }
    }


def apply_post_processing(
    records,
    protected_attr="Gender",
    algorithm="calibrated_eq_odds",
    label_col="decision"
):
    """
    Apply AIF360 post-processing to correct biased decisions.

    Algorithms:
        - "calibrated_eq_odds": CalibratedEqOddsPostprocessing (default)
        - "eq_odds": EqOddsPostprocessing
        - "reject_option": RejectOptionClassification

    Returns:
        dict with original_metrics, corrected_metrics, corrected_decisions, algorithm_used
    """
    if not AIF360_AVAILABLE:
        return {"success": False, "error": "AIF360 not installed. Install with: pip install aif360"}

    dataset, privileged, unprivileged = _build_dataset(records, protected_attr, label_col)

    if dataset is None:
        return {
            "success": False,
            "error": "Not enough data for post-processing. Need records with both privileged and unprivileged groups.",
            "original_metrics": None,
            "corrected_metrics": None,
        }

    # --- Compute ORIGINAL metrics ---
    orig_metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )
    orig_di = float(orig_metric.disparate_impact())
    orig_spd = float(orig_metric.statistical_parity_difference())
    if np.isinf(orig_di) or np.isnan(orig_di):
        orig_di = 0.0
    if np.isinf(orig_spd) or np.isnan(orig_spd):
        orig_spd = 0.0

    # --- Apply post-processing ---
    # For post-processing, we need "predicted" labels and "actual" labels.
    # In our case, the AI's decisions ARE the predictions.
    # We treat them as both the prediction and the "ground truth" for the purpose
    # of the post-processing correction (since we don't have separate ground truth).
    # The post-processor will adjust the predictions to be fairer.

    dataset_pred = dataset.copy(deepcopy=True)

    try:
        if algorithm == "calibrated_eq_odds":
            postprocessor = CalibratedEqOddsPostprocessing(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                cost_constraint="weighted",  # balanced between FPR and FNR
                seed=42
            )
            # Fit on the original dataset (as "true" labels) and predicted dataset
            corrected_dataset = postprocessor.fit_predict(dataset, dataset_pred)

        elif algorithm == "eq_odds":
            postprocessor = EqOddsPostprocessing(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                seed=42
            )
            corrected_dataset = postprocessor.fit_predict(dataset, dataset_pred)

        elif algorithm == "reject_option":
            postprocessor = RejectOptionClassification(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                low_class_thresh=0.01,
                high_class_thresh=0.99,
                num_class_thresh=100,
                num_ROC_margin=50,
                metric_name="Statistical parity difference",
                metric_ub=0.05,
                metric_lb=-0.05
            )
            corrected_dataset = postprocessor.fit_predict(dataset, dataset_pred)

        else:
            return {
                "success": False,
                "error": f"Unknown algorithm: {algorithm}. Use 'calibrated_eq_odds', 'eq_odds', or 'reject_option'.",
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Post-processing failed: {str(e)}",
            "original_metrics": {
                "disparate_impact": round(orig_di, 4),
                "statistical_parity_diff": round(orig_spd, 4),
            }
        }

    # --- Compute CORRECTED metrics ---
    corr_metric = BinaryLabelDatasetMetric(
        corrected_dataset,
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )
    corr_di = float(corr_metric.disparate_impact())
    corr_spd = float(corr_metric.statistical_parity_difference())
    if np.isinf(corr_di) or np.isnan(corr_di):
        corr_di = 0.0
    if np.isinf(corr_spd) or np.isnan(corr_spd):
        corr_spd = 0.0

    # --- Extract corrected decisions ---
    corrected_labels = corrected_dataset.labels.flatten().tolist()
    original_labels = dataset.labels.flatten().tolist()

    # Count how many decisions were flipped
    flipped_count = sum(
        1 for o, c in zip(original_labels, corrected_labels) if o != c
    )

    # Map corrected labels back to decision strings
    reverse_map = {1: "Favorable", 0: "Unfavorable"}
    corrected_decisions = [reverse_map.get(int(l), "Unknown") for l in corrected_labels]

    return {
        "success": True,
        "algorithm_used": algorithm,
        "algorithm_description": _get_algorithm_description(algorithm),
        "original_metrics": {
            "disparate_impact": round(orig_di, 4),
            "statistical_parity_diff": round(orig_spd, 4),
            "bias_detected": orig_di < 0.8 or orig_di > 1.25,
        },
        "corrected_metrics": {
            "disparate_impact": round(corr_di, 4),
            "statistical_parity_diff": round(corr_spd, 4),
            "bias_detected": corr_di < 0.8 or corr_di > 1.25,
        },
        "improvement": {
            "disparate_impact_change": round(corr_di - orig_di, 4),
            "spd_change": round(abs(corr_spd) - abs(orig_spd), 4),
            "decisions_flipped": flipped_count,
            "total_decisions": len(original_labels),
            "flip_percentage": round(flipped_count / max(len(original_labels), 1) * 100, 1),
        },
        "corrected_decisions": corrected_decisions,
        "protected_attribute": protected_attr,
    }


def get_recommendation(records, protected_attr="Gender"):
    """
    Analyze the bias and recommend which post-processing algorithm to use.
    """
    metrics = compute_bias_metrics(records, protected_attr)

    if not metrics.get("bias_detected", False):
        return {
            "recommendation": "none",
            "reason": "No significant bias detected. Disparate impact is within fair range [0.80, 1.25].",
            "metrics": metrics,
        }

    di = metrics.get("disparate_impact", 1.0)
    severity = metrics.get("severity", "none")

    if severity == "high":
        return {
            "recommendation": "eq_odds",
            "reason": (
                f"Severe bias detected (DI={di}). Recommending Equalized Odds — "
                "the strictest algorithm that forces equal true positive and false positive rates."
            ),
            "metrics": metrics,
        }
    elif di < 0.8:
        return {
            "recommendation": "calibrated_eq_odds",
            "reason": (
                f"Moderate negative bias detected (DI={di}). Recommending Calibrated Equalized Odds — "
                "balances fairness with accuracy, best for moderate bias correction."
            ),
            "metrics": metrics,
        }
    else:
        return {
            "recommendation": "reject_option",
            "reason": (
                f"Positive bias detected (DI={di}). Recommending Reject Option Classification — "
                "flips borderline decisions to reduce over-favoring."
            ),
            "metrics": metrics,
        }


def _get_algorithm_description(algorithm):
    """Return a human-readable description of the algorithm."""
    descriptions = {
        "calibrated_eq_odds": (
            "Calibrated Equalized Odds: Adjusts prediction probabilities to equalize "
            "true and false positive rates across privileged and unprivileged groups, "
            "while minimizing changes to the original predictions."
        ),
        "eq_odds": (
            "Equalized Odds: The strictest post-processing method. Forces equal "
            "true positive rate (TPR) and false positive rate (FPR) across groups. "
            "May flip more decisions but achieves maximum fairness."
        ),
        "reject_option": (
            "Reject Option Classification: Identifies decisions near the classification "
            "boundary and flips them in favor of the unprivileged group. Minimal impact "
            "on confident decisions, targets only borderline cases."
        ),
    }
    return descriptions.get(algorithm, "Unknown algorithm")
