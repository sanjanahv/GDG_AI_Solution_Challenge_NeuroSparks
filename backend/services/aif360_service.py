"""
AIF360 Bias Detection & Post-Processing Service.

Uses IBM's AI Fairness 360 library to:
  1. Compute bias metrics (Disparate Impact, Statistical Parity Difference)
  2. Apply post-processing algorithms to de-bias decisions (Reweighing, CalibratedEqOdds)

AIF360 works on BATCHES (datasets), not single decisions.
The frontend stores decisions in-session; after 5+ decisions we can run analysis.

Key metric: Disparate Impact
  - < 0.8 → Negative bias against unprivileged group
  - > 1.25 → Positive bias (over-favoring unprivileged)
  - 0.8 to 1.25 → Acceptable range (US 80% employment law rule)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Decision label mappings across domains
DECISION_MAP = {
    "hire": 1, "reject": 0,
    "approve": 1, "deny": 0,
    "admit": 1, "waitlist": 0,
    # Numeric pass-through
    1: 1, 0: 0, "1": 1, "0": 0
}

# Protected attributes to check per domain
DOMAIN_PROTECTED_ATTRS = {
    "job": ["Gender", "Age"],
    "loan": ["Gender", "Race"],
    "college": ["Gender", "Parent Income"]
}


def _encode_decision(decision_value) -> int:
    """Convert a decision string to binary 0/1."""
    if isinstance(decision_value, (int, float)):
        return int(decision_value)
    return DECISION_MAP.get(str(decision_value).lower().strip(), 0)


def _encode_protected_attr(series: pd.Series) -> pd.Series:
    """
    Encode a protected attribute column to numeric.
    For string columns (e.g., gender), use label encoding.
    """
    if series.dtype in [np.float64, np.int64, float, int]:
        # Already numeric — bin if needed (e.g., age → age_group)
        median = series.median()
        return (series >= median).astype(int)

    # String encoding: most common value = 1 (privileged), others = 0
    mode_val = series.mode()[0] if len(series.mode()) > 0 else series.iloc[0]
    return (series == mode_val).astype(int)


def compute_bias_metrics(records: list, protected_attr: str, label_col: str = "decision") -> dict:
    """
    Compute AIF360 bias metrics on a batch of decision records.

    Args:
        records: List of dicts, each containing profile attributes + decision field
        protected_attr: The protected attribute column name to check (e.g., "Gender")
        label_col: The column containing the decision (default: "decision")

    Returns:
        dict with disparate_impact, statistical_parity_diff, bias_detected, sample_size
    """
    if len(records) < 3:
        return {
            "disparate_impact": 1.0,
            "statistical_parity_diff": 0.0,
            "bias_detected": False,
            "protected_attribute": protected_attr,
            "sample_size": len(records),
            "message": "Need at least 3 decisions to compute meaningful bias metrics."
        }

    try:
        df = pd.DataFrame(records)

        # Ensure the protected attribute and decision columns exist
        if protected_attr not in df.columns:
            return {
                "disparate_impact": 1.0,
                "statistical_parity_diff": 0.0,
                "bias_detected": False,
                "protected_attribute": protected_attr,
                "sample_size": len(records),
                "message": f"Protected attribute '{protected_attr}' not found in records."
            }

        # Encode decision to binary
        df["_label"] = df[label_col].apply(_encode_decision)

        # Encode protected attribute to binary
        df["_protected"] = _encode_protected_attr(df[protected_attr])

        # Check we have both groups
        unique_groups = df["_protected"].nunique()
        if unique_groups < 2:
            return {
                "disparate_impact": 1.0,
                "statistical_parity_diff": 0.0,
                "bias_detected": False,
                "protected_attribute": protected_attr,
                "sample_size": len(records),
                "message": "Only one group found — need diversity in protected attribute to measure bias."
            }

        # Compute metrics manually (avoids AIF360 import issues on some systems)
        # Privileged group = _protected == 1, Unprivileged = _protected == 0
        priv = df[df["_protected"] == 1]
        unpriv = df[df["_protected"] == 0]

        # Positive outcome rates
        priv_rate = priv["_label"].mean() if len(priv) > 0 else 0
        unpriv_rate = unpriv["_label"].mean() if len(unpriv) > 0 else 0

        # Disparate Impact = P(Y=1 | unprivileged) / P(Y=1 | privileged)
        if priv_rate == 0:
            disparate_impact = float('inf') if unpriv_rate > 0 else 1.0
        else:
            disparate_impact = round(unpriv_rate / priv_rate, 4)

        # Statistical Parity Difference = P(Y=1 | unprivileged) - P(Y=1 | privileged)
        stat_parity_diff = round(unpriv_rate - priv_rate, 4)

        # Bias detected if outside 80% rule range
        bias_detected = disparate_impact < 0.8 or disparate_impact > 1.25

        return {
            "disparate_impact": disparate_impact,
            "statistical_parity_diff": stat_parity_diff,
            "bias_detected": bias_detected,
            "protected_attribute": protected_attr,
            "sample_size": len(records),
            "privileged_positive_rate": round(priv_rate, 4),
            "unprivileged_positive_rate": round(unpriv_rate, 4),
            "privileged_count": len(priv),
            "unprivileged_count": len(unpriv)
        }

    except Exception as e:
        logger.error(f"Error computing bias metrics: {e}")
        return {
            "disparate_impact": 1.0,
            "statistical_parity_diff": 0.0,
            "bias_detected": False,
            "protected_attribute": protected_attr,
            "sample_size": len(records),
            "message": f"Error computing metrics: {str(e)}"
        }


def compute_bias_metrics_aif360(records: list, protected_attr: str, label_col: str = "decision") -> dict:
    """
    Full AIF360 implementation using BinaryLabelDataset.
    Falls back to manual computation if AIF360 is not installed.
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric

        df = pd.DataFrame(records)
        df[label_col] = df[label_col].apply(_encode_decision)
        df[protected_attr] = _encode_protected_attr(df[protected_attr])

        dataset = BinaryLabelDataset(
            df=df[[protected_attr, label_col]],
            label_names=[label_col],
            protected_attribute_names=[protected_attr],
            favorable_label=1,
            unfavorable_label=0
        )

        privileged = [{protected_attr: 1}]
        unprivileged = [{protected_attr: 0}]

        metric = BinaryLabelDatasetMetric(
            dataset,
            privileged_groups=privileged,
            unprivileged_groups=unprivileged
        )

        di = metric.disparate_impact()
        spd = metric.statistical_parity_difference()

        return {
            "disparate_impact": round(di, 4) if not np.isinf(di) else 999.0,
            "statistical_parity_diff": round(spd, 4),
            "bias_detected": di < 0.8 or di > 1.25,
            "protected_attribute": protected_attr,
            "sample_size": len(records)
        }

    except ImportError:
        logger.warning("AIF360 not installed, falling back to manual computation.")
        return compute_bias_metrics(records, protected_attr, label_col)


def apply_reweighing(records: list, protected_attr: str, label_col: str = "decision") -> dict:
    """
    Apply AIF360 Reweighing algorithm to de-bias the dataset.
    Returns the instance weights that should be applied.
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.preprocessing import Reweighing

        df = pd.DataFrame(records)
        df[label_col] = df[label_col].apply(_encode_decision)
        df[protected_attr] = _encode_protected_attr(df[protected_attr])

        dataset = BinaryLabelDataset(
            df=df[[protected_attr, label_col]],
            label_names=[label_col],
            protected_attribute_names=[protected_attr],
            favorable_label=1,
            unfavorable_label=0
        )

        rw = Reweighing(
            privileged_groups=[{protected_attr: 1}],
            unprivileged_groups=[{protected_attr: 0}]
        )
        transformed = rw.fit_transform(dataset)

        # Compute new metrics after reweighing
        from aif360.metrics import BinaryLabelDatasetMetric
        metric = BinaryLabelDatasetMetric(
            transformed,
            privileged_groups=[{protected_attr: 1}],
            unprivileged_groups=[{protected_attr: 0}]
        )

        return {
            "algorithm": "reweighing",
            "original_weights": dataset.instance_weights.tolist(),
            "corrected_weights": transformed.instance_weights.tolist(),
            "new_disparate_impact": round(metric.disparate_impact(), 4),
            "new_statistical_parity_diff": round(metric.statistical_parity_difference(), 4),
            "success": True
        }

    except ImportError:
        return {
            "algorithm": "reweighing",
            "success": False,
            "message": "AIF360 not installed. Install with: pip install aif360"
        }
    except Exception as e:
        return {
            "algorithm": "reweighing",
            "success": False,
            "message": str(e)
        }


def get_domain_protected_attrs(domain: str) -> list[str]:
    """Get the list of protected attributes to check for a domain."""
    return DOMAIN_PROTECTED_ATTRS.get(domain.lower(), ["Gender"])
