"""Coverage smoke test for addie-models — one molecule, all fields.

Why this test exists
--------------------
The TDC overlay loader can silently drop endpoints (e.g. from a wrong
weights-key prefix) while `/health` still reports all models loaded — the
endpoints just come back null in the response. This test makes that gap loud.

It asserts every field a healthy service should produce for a representative
molecule (aspirin). Run it after any change to:

  - MODEL_LIST (main.py)
  - TDC_ENDPOINTS / ADDIE_REPLACEMENT_MAP (tdc_sota_integration.py)
  - TDC_SOTA_KEY_PREFIX / MODEL_PREFIX / weights layout
  - the response serializer or aggregate `*_score` calculations

How to run
----------
Start the service (see the README), then point this test at it:

    ADDIE_URL=http://localhost:8025 python3 tests/test_model_coverage.py

The test exits non-zero on any missing or malformed field and prints
a concise per-category diff.
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Expected output schema. Update alongside MODEL_LIST / TDC additions.
# ---------------------------------------------------------------------------

# Base ADDIE-derived response fields (one or more per loaded ADDIE model).
ADDIE_FIELDS = {
    # affinity
    "binding_affinity_score",
    # cardiotoxicity (4 time-windows + max aggregate)
    "cardiotoxicity_1d_probability",
    "cardiotoxicity_5d_probability",
    "cardiotoxicity_10d_probability",
    "cardiotoxicity_30d_probability",
    "cardiotoxicity_max_probability",
    # CYP450 inhibition
    "cyp1a2_inhibitor_probability",
    "cyp2c19_inhibitor_probability",
    "cyp2c9_inhibitor_probability",
    "cyp2d6_inhibitor_probability",
    "cyp3a4_inhibitor_probability",
    # Nuclear receptors
    "nr_ahr_agonist_probability",
    "nr_ar_lbd_agonist_probability",
    "nr_ar_agonist_probability",
    "nr_aromatase_inhibitor_probability",
    "nr_er_lbd_agonist_probability",
    "nr_er_agonist_probability",
    "nr_ppar_gamma_agonist_probability",
    # Stress response
    "sr_are_activation_probability",
    "sr_atad5_activation_probability",
    "sr_hse_activation_probability",
    "sr_mmp_activation_probability",
    "sr_p53_activation_probability",
    # Toxicity
    "ames_mutagenicity_probability",
    "carcinogenicity_probability",
    "clinical_toxicity_probability",
    "developmental_toxicity_probability",
    "eye_corrosion_probability",
    "eye_irritation_probability",
    "hepatotoxicity_probability",
    "reproductive_toxicity_probability",
    "respiratory_toxicity_probability",
}

# Fields the TDC SOTA overlay introduces (no ADDIE equivalent).
# These are surfaced as raw values in the response.
TDC_EXCLUSIVE_FIELDS = {
    "caco2_permeability",
    "hia_probability",
    "pgp_inhibitor_probability",   # this one has a base too; kept here for clarity
    "pgp_substrate_probability",
    "bioavailability_probability",
    "lipophilicity_log_ratio",
    "aqueous_solubility_log_mol_L",
    "bbb_penetration_probability",
    "ppbr_percent",
    "vdss_L_kg",
    "cyp2c9_substrate_probability",
    "cyp2d6_substrate_probability",
    "cyp3a4_substrate_probability",
    "half_life_hr",
    "clearance_hepatocyte",
    "clearance_microsome",
    "ld50_log_mol_kg",
    "herg_blocker_probability",
}

# Aggregate scores computed downstream from the raw fields.
AGGREGATE_FIELDS = {
    "absorption_score",
    "distribution_score",
    "metabolism_score",
    "overall_toxicity_score",
    "cyp_inhibition_risk_score",
    "cyp_substrate_max_probability",
    "cns_assessment",
}

ALL_EXPECTED_FIELDS = ADDIE_FIELDS | TDC_EXCLUSIVE_FIELDS | AGGREGATE_FIELDS


# ---------------------------------------------------------------------------
# Field-level sanity ranges. Any field present that violates these fails the
# test — guards against the case where the model loads but returns NaN /
# wildly out-of-distribution values.
# ---------------------------------------------------------------------------

PROBABILITY_FIELDS = {f for f in ALL_EXPECTED_FIELDS if f.endswith("_probability")}

RANGE_CHECKS = {
    # probability fields — [0, 1] strict
    **{f: (0.0, 1.0) for f in PROBABILITY_FIELDS},
    # 0-100 scores
    "absorption_score": (0.0, 1.0),
    "distribution_score": (0.0, 1.0),
    "metabolism_score": (0.0, 1.0),
    "overall_toxicity_score": (0.0, 1.0),
    "cyp_inhibition_risk_score": (0.0, 1.0),
    "cyp_substrate_max_probability": (0.0, 1.0),
    "cardiotoxicity_max_probability": (0.0, 1.0),
    # rough plausibility checks on raw outputs
    "caco2_permeability": (-10.0, 0.0),       # log cm/s; small org molecules ~ -6 to -4
    "lipophilicity_log_ratio": (-5.0, 10.0),
    "aqueous_solubility_log_mol_L": (-15.0, 5.0),
    "ppbr_percent": (0.0, 100.0),
    "vdss_L_kg": (0.0, 100.0),
    "half_life_hr": (0.0, 200.0),
    "clearance_hepatocyte": (0.0, 5000.0),
    "clearance_microsome": (0.0, 5000.0),
    "ld50_log_mol_kg": (0.0, 10.0),
    "binding_affinity_score": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Test entrypoint
# ---------------------------------------------------------------------------

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Local service URL by default; override with the ADDIE_URL env var.
SERVICE_URL = os.environ.get(
    "ADDIE_URL",
    "http://localhost:8025",
)
API_KEY = os.environ.get("API_KEY") or os.environ.get("ADDIE_API_KEY", "")


def call_process() -> dict:
    payload = {
        "molecules": [{"smiles": ASPIRIN_SMILES, "id": "aspirin"}],
        "include_descriptors": False,
        "include_confidence": False,
        "enhanced_mode": True,
    }
    req = urllib.request.Request(
        f"{SERVICE_URL}/addie/process",
        data=json.dumps(payload).encode(),
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    return json.loads(body)


def main() -> int:
    print(f"POST {SERVICE_URL}/addie/process  smiles={ASPIRIN_SMILES}")
    try:
        data = call_process()
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read().decode()[:300]}")
        return 2
    except Exception as e:
        print(f"call failed: {type(e).__name__}: {e}")
        return 2

    results = data.get("results") or []
    if not results:
        print("response had no `results` array")
        return 3
    preds = results[0].get("predictions") or {}
    if not preds:
        print("results[0].predictions is empty")
        return 3

    present = set(preds.keys())
    missing = sorted(ALL_EXPECTED_FIELDS - present)
    extra = sorted(present - ALL_EXPECTED_FIELDS)

    # Range checks
    bad_values = []
    for f, (lo, hi) in RANGE_CHECKS.items():
        if f not in preds:
            continue
        v = preds[f]
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            bad_values.append((f, v, "null/nan/inf"))
            continue
        if not isinstance(v, (int, float)):
            bad_values.append((f, v, f"non-numeric ({type(v).__name__})"))
            continue
        if not (lo <= v <= hi):
            bad_values.append((f, v, f"out of [{lo}, {hi}]"))

    # Report
    print()
    print(f"=== fields present:  {len(present)}/{len(ALL_EXPECTED_FIELDS)} expected ===")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for f in missing:
            print(f"  - {f}")
    if extra:
        print(f"extra ({len(extra)}, informational):")
        for f in extra[:10]:
            print(f"  + {f}")
        if len(extra) > 10:
            print(f"  ... and {len(extra) - 10} more")
    if bad_values:
        print(f"BAD VALUES ({len(bad_values)}):")
        for f, v, why in bad_values:
            print(f"  - {f} = {v}  ({why})")

    health = data.get("model_count")
    if health is not None:
        print(f"\nresponse model_count: {health}")

    if missing or bad_values:
        print("\nFAIL")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
