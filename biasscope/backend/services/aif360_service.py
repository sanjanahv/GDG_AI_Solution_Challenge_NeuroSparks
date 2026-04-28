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
            str_val = str(value).strip()
            # Handle age ranges like "25-34" by averaging
            if '-' in str_val and not str_val.startswith('-'):
                parts = str_val.split('-')
                age = int((int(parts[0].strip()) + int(parts[1].strip())) / 2)
            else:
                # Handle plain numbers, possibly with '+' suffix like "65+"
                age = int(str_val.rstrip('+'))
            return 1 if age < 40 else 0  # Young=1 (privileged), Old=0
        except (ValueError, TypeError, IndexError):
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

    FIX: The original code used identical datasets for ground truth and
    predictions, which caused the post-processor to either do nothing or
    crash. This version synthesizes a "fair ground truth" where favorable
    outcomes are distributed equally across demographic groups.

    Returns:
        dict with original_metrics, corrected_metrics, corrected_decisions, algorithm_used
    """
    if not AIF360_AVAILABLE:
        # Manual correction fallback — flip decisions to equalize rates
        return _manual_correction_fallback(records, protected_attr)

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

    # --- Synthesize a FAIR ground truth ---
    # The key insight: AIF360 post-processing needs ground truth != predictions.
    # Since we have no real ground truth, we create one where the overall positive
    # rate is preserved but distributed EQUALLY across demographic groups.
    # This tells the post-processor: "here's what fair decisions would look like."

    dataset_true = dataset.copy(deepcopy=True)
    dataset_pred = dataset.copy(deepcopy=True)

    df = dataset.convert_to_dataframe()[0]
    overall_positive_rate = df[label_col].mean()

    # Create fair labels: each group gets the overall positive rate
    fair_labels = np.zeros(len(df))
    for group_val in [0, 1]:
        group_mask = (df[protected_attr] == group_val).values
        group_size = group_mask.sum()
        if group_size == 0:
            continue
        n_positive = max(1, int(round(overall_positive_rate * group_size)))
        n_positive = min(n_positive, group_size)

        # Assign positive labels to first n_positive records in this group
        group_indices = np.where(group_mask)[0]
        # Shuffle to avoid order bias
        rng = np.random.RandomState(42)
        rng.shuffle(group_indices)
        fair_labels[group_indices[:n_positive]] = 1.0

    dataset_true.labels = fair_labels.reshape(-1, 1)

    # Add synthetic scores (confidence) to enable reject_option to work
    # Use a simple distance-from-threshold as proxy for confidence
    scores = np.where(dataset_pred.labels == 1, 0.7, 0.3)
    # Add noise to create a "borderline" zone
    rng = np.random.RandomState(42)
    scores = scores + rng.uniform(-0.2, 0.2, size=scores.shape)
    scores = np.clip(scores, 0.01, 0.99)
    dataset_pred.scores = scores.reshape(-1, 1)

    try:
        if algorithm == "calibrated_eq_odds":
            postprocessor = CalibratedEqOddsPostprocessing(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                cost_constraint="weighted",
                seed=42
            )
            corrected_dataset = postprocessor.fit_predict(dataset_true, dataset_pred)

        elif algorithm == "eq_odds":
            postprocessor = EqOddsPostprocessing(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                seed=42
            )
            corrected_dataset = postprocessor.fit_predict(dataset_true, dataset_pred)

        elif algorithm == "reject_option":
            postprocessor = RejectOptionClassification(
                privileged_groups=privileged,
                unprivileged_groups=unprivileged,
                low_class_thresh=0.2,
                high_class_thresh=0.8,
                num_class_thresh=100,
                num_ROC_margin=50,
                metric_name="Statistical parity difference",
                metric_ub=0.05,
                metric_lb=-0.05
            )
            corrected_dataset = postprocessor.fit_predict(dataset_true, dataset_pred)

        else:
            return {
                "success": False,
                "error": f"Unknown algorithm: {algorithm}. Use 'calibrated_eq_odds', 'eq_odds', or 'reject_option'.",
            }

    except Exception as e:
        # If AIF360 post-processing still fails, use manual correction
        manual_result = _manual_correction_fallback(records, protected_attr)
        manual_result["algorithm_note"] = f"AIF360 {algorithm} failed ({str(e)}), used manual equalization instead."
        manual_result["original_metrics"] = {
            "disparate_impact": round(orig_di, 4),
            "statistical_parity_diff": round(orig_spd, 4),
        }
        return manual_result

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
    original_labels = dataset_pred.labels.flatten().tolist()

    flipped_count = sum(
        1 for o, c in zip(original_labels, corrected_labels) if o != c
    )

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


def _manual_correction_fallback(records, protected_attr):
    """
    Manual bias correction when AIF360 is not available.
    Equalizes positive outcome rates across demographic groups by
    flipping the minimum number of decisions needed.
    """
    # Group records by protected attribute value
    groups = {}  # group_val -> [list of (index, decision_val)]
    for i, r in enumerate(records):
        attrs = r.get("attributes", r.get("profile", r))
        pv = None
        for k, v in attrs.items():
            if k.lower() == protected_attr.lower():
                pv = v
                break
        if pv is None:
            continue
        enc = _encode_protected_attribute(pv, protected_attr)
        dec_raw = r.get("decision", "")
        dec = DECISION_MAP.get(str(dec_raw), 0) if isinstance(dec_raw, str) else int(dec_raw)
        groups.setdefault(enc, []).append((i, dec))

    if len(groups) < 2:
        return {
            "success": False,
            "error": "Need both privileged and unprivileged groups for correction.",
        }

    # Compute rates
    rates = {}
    for gv, entries in groups.items():
        pos = sum(d for _, d in entries)
        rates[gv] = pos / len(entries) if entries else 0

    # Target: overall positive rate
    all_decisions = [d for entries in groups.values() for _, d in entries]
    target_rate = sum(all_decisions) / len(all_decisions) if all_decisions else 0.5

    # Flip decisions in each group to match target rate
    corrected = list(all_decisions)
    corrected_map = {}  # original_index -> new_decision
    flipped = 0

    for gv, entries in groups.items():
        current_pos = sum(d for _, d in entries)
        target_pos = max(0, min(len(entries), int(round(target_rate * len(entries)))))
        diff = target_pos - current_pos

        if diff > 0:
            # Need more positives — flip negatives to positive
            negatives = [(idx, d) for idx, d in entries if d == 0]
            for idx, _ in negatives[:diff]:
                corrected_map[idx] = 1
                flipped += 1
        elif diff < 0:
            # Need fewer positives — flip positives to negative
            positives = [(idx, d) for idx, d in entries if d == 1]
            for idx, _ in positives[:abs(diff)]:
                corrected_map[idx] = 0
                flipped += 1

    # Build corrected decisions list
    corrected_decisions = []
    for i, r in enumerate(records):
        if i in corrected_map:
            corrected_decisions.append("Favorable" if corrected_map[i] == 1 else "Unfavorable")
        else:
            dec_raw = r.get("decision", "")
            dec = DECISION_MAP.get(str(dec_raw), 0) if isinstance(dec_raw, str) else int(dec_raw)
            corrected_decisions.append("Favorable" if dec == 1 else "Unfavorable")

    # Compute corrected metrics
    priv_pos = priv_tot = unpriv_pos = unpriv_tot = 0
    for gv, entries in groups.items():
        for idx, _ in entries:
            new_dec = corrected_map.get(idx, DECISION_MAP.get(str(records[idx].get("decision", "")), 0))
            if gv == 1:
                priv_tot += 1
                priv_pos += new_dec
            else:
                unpriv_tot += 1
                unpriv_pos += new_dec

    pr = priv_pos / max(priv_tot, 1)
    ur = unpriv_pos / max(unpriv_tot, 1)

    # Handle DI edge cases properly instead of masking with 0.001
    if pr == 0 and ur == 0:
        corr_di = 1.0  # Both groups have 0% positive rate — perfectly "equal"
    elif pr == 0:
        corr_di = float('inf')  # Unprivileged has positive, privileged has 0
    else:
        corr_di = ur / pr

    # Clamp to a reportable range
    if np.isinf(corr_di) or np.isnan(corr_di):
        corr_di = 0.0

    return {
        "success": True,
        "algorithm_used": "manual_equalization",
        "algorithm_description": (
            "Manual Equalization: Flips the minimum number of decisions needed "
            "to equalize positive outcome rates across demographic groups."
        ),
        "original_metrics": {
            "disparate_impact": None,
            "statistical_parity_diff": None,
            "bias_detected": True,
        },
        "corrected_metrics": {
            "disparate_impact": round(corr_di, 4),
            "statistical_parity_diff": round(ur - pr, 4),
            "bias_detected": corr_di < 0.8 or corr_di > 1.25,
        },
        "improvement": {
            "decisions_flipped": flipped,
            "total_decisions": len(records),
            "flip_percentage": round(flipped / max(len(records), 1) * 100, 1),
        },
        "corrected_decisions": corrected_decisions,
        "protected_attribute": protected_attr,
        "method": "manual_fallback",
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
