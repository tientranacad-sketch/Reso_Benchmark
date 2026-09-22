import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


# ---------------------------------------------------------------------------
# 1. Whitelist In-Memory Loader
# ---------------------------------------------------------------------------
def load_whitelist_camelcase(filepath: str = "reso_fields.txt") -> Tuple[Set[str], Dict[str, str]]:
    """
    Loads whitelist fields, converts PascalCase to camelCase in memory without modifying
    the original source file, and returns a lowercase lookup set and a case-mapping dict.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Whitelist file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Transform PascalCase to camelCase in memory
    camel_fields = [field[0].lower() + field[1:] for field in lines if field]
    allowed_fields_lower = {f.lower() for f in camel_fields}
    original_camel_map = {f.lower(): f for f in camel_fields}

    logging.info(f"Loaded whitelist from '{filepath}' ({len(camel_fields)} fields).")
    return allowed_fields_lower, original_camel_map


# ---------------------------------------------------------------------------
# 2. JSON Cleaning & Whitelist Filtering Utilities
# ---------------------------------------------------------------------------
def clean_json_str(raw_str: Any) -> str:
    """Strips markdown code blocks and whitespace from raw JSON strings."""
    if not isinstance(raw_str, str) or pd.isna(raw_str):
        return "{}"
    raw_str = raw_str.strip()
    if raw_str.startswith("```"):
        raw_str = re.sub(r"^```(?:json)?\s*", "", raw_str, flags=re.IGNORECASE)
        raw_str = re.sub(r"\s*```$", "", raw_str)
    return raw_str.strip()


def filter_whitelist_json(
    json_obj: Any,
    allowed_fields_lower: Set[str],
    original_camel_map: Dict[str, str]
) -> Any:
    """Recursively filters JSON objects to retain only keys present in the RESO whitelist."""
    if isinstance(json_obj, dict):
        filtered = {}
        for k, v in json_obj.items():
            k_lower = str(k).lower()
            if k_lower in allowed_fields_lower:
                standard_camel_key = original_camel_map[k_lower]
                filtered[standard_camel_key] = filter_whitelist_json(
                    v, allowed_fields_lower, original_camel_map
                )
        return filtered
    elif isinstance(json_obj, list):
        return [
            filter_whitelist_json(item, allowed_fields_lower, original_camel_map)
            for item in json_obj
        ]
    else:
        return json_obj


# ---------------------------------------------------------------------------
# 3. Differentiation Score (DS) Metric Computation
# ---------------------------------------------------------------------------
def calculate_ds_score(
    json_gen_raw: Any,
    allowed_fields_lower: Set[str],
    original_camel_map: Dict[str, str]
) -> Tuple[float, str, str]:
    """
    Computes Differentiation Score (DS) between generated JSON and whitelist-filtered JSON.
    DS = 1.0 - TF-IDF character n-gram cosine similarity.

    - DS = 0.0: Perfect compliance (zero un-whitelisted schema keys).
    - DS = 1.0: Complete divergence (all fields lie outside the whitelist).
    """
    cleaned_gen = clean_json_str(json_gen_raw)

    try:
        parsed_gen = json.loads(cleaned_gen)
    except Exception:
        return 1.0, cleaned_gen, "{}"

    json_gen_min = json.dumps(parsed_gen, separators=(",", ":"), ensure_ascii=False)

    parsed_white = filter_whitelist_json(parsed_gen, allowed_fields_lower, original_camel_map)
    json_white_min = json.dumps(parsed_white, separators=(",", ":"), ensure_ascii=False)

    if json_gen_min == json_white_min:
        return 0.0, json_gen_min, json_white_min
    if json_white_min in ["{}", "[]", "null"] and json_gen_min not in ["{}", "[]", "null"]:
        return 1.0, json_gen_min, json_white_min

    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
        tfidf_matrix = vectorizer.fit_transform([json_gen_min, json_white_min])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        ds = 1.0 - float(sim)
        return float(np.clip(ds, 0.0, 1.0)), json_gen_min, json_white_min
    except Exception:
        return 1.0, json_gen_min, json_white_min


# ---------------------------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------------------------
def run_ds_evaluation(
    input_file: str,
    whitelist_file: str,
    output_detail_file: str,
    output_summary_file: str
) -> None:
    """Runs full Differentiation Score (DS) evaluation pipeline."""
    allowed_fields_lower, original_camel_map = load_whitelist_camelcase(whitelist_file)

    if not os.path.exists(input_file):
        logging.error(f"Input file not found: '{input_file}'")
        return

    df = pd.read_csv(input_file)
    logging.info(f"Loaded '{input_file}' containing {len(df)} records.")

    # Validation check: DS Evaluation requires postprocess.py output
    if "cleaned_json" not in df.columns:
        logging.warning(
            "⚠️ CRITICAL WARNING: Column 'cleaned_json' not found in input file! "
            "Differentiation Score (DS) strictly requires preprocessed datasets from 'postprocess.py' "
            "to ensure accurate schema parsing. Falling back to 'generated_json' may produce skewed scores."
        )

    logging.info("Calculating Differentiation Score (DS)...")
    ds_scores, json_gens, json_whites = [], [], []

    for idx, row in df.iterrows():
        # Prefer cleaned_json from postprocess.py, fallback to raw generated_json
        json_val = row.get("cleaned_json")
        if pd.isna(json_val) or not str(json_val).strip() or str(json_val).strip() == "{}":
            json_val = row.get("generated_json")

        ds, j_gen, j_white = calculate_ds_score(json_val, allowed_fields_lower, original_camel_map)
        ds_scores.append(ds)
        json_gens.append(j_gen)
        json_whites.append(j_white)

    df["ds_score"] = ds_scores
    df["json_gen_min"] = json_gens
    df["json_white_min"] = json_whites

    # Save detailed row-level results
    df.to_csv(output_detail_file, index=False, encoding="utf-8-sig")
    logging.info(f"Detailed DS results saved to: '{output_detail_file}'")

    # Aggregate metric summary by model
    summary_model = df.groupby("model")["ds_score"].agg(
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    ).reset_index().sort_values(by="Mean", ascending=True)

    summary_model.to_csv(output_summary_file, index=False, encoding="utf-8-sig")
    logging.info(f"DS summary report saved to: '{output_summary_file}'")

    logging.info("\n" + "=" * 60)
    logging.info(" DIFFERENTIATION SCORE (DS) SUMMARY BY MODEL ")
    logging.info("=" * 60)
    print(summary_model.to_string(index=False))
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# 5. CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Differentiation Score (DS) Metric Evaluation Pipeline (Requires postprocessed dataset from postprocess.py)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_cleaned_unified.csv",
        help="Path to post-processed CSV file from postprocess.py (must contain 'cleaned_json')"
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default="reso_fields.txt",
        help="Path to RESO whitelist text file"
    )
    parser.add_argument(
        "--output-detail",
        type=str,
        default="benchmark_cleaned_unified_with_ds.csv",
        help="Path to save row-level detailed DS score CSV"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="benchmark_ds_summary.csv",
        help="Path to save aggregated DS summary report CSV"
    )

    args = parser.parse_args()

    run_ds_evaluation(
        input_file=args.input,
        whitelist_file=args.whitelist,
        output_detail_file=args.output_detail,
        output_summary_file=args.output_summary
    )


if __name__ == "__main__":
    main()
