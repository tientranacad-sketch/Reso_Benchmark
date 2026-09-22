import argparse
import json
import logging
import os
import re
from typing import Any, Dict, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


# ---------------------------------------------------------------------------
# 1. RESO Standard Fields Loader
# ---------------------------------------------------------------------------
def load_reso_standard_fields(file_path: str = "reso_fields.txt") -> Set[str]:
    """Loads target RESO 2.0 standard fields from whitelist text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RESO fields whitelist file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        fields = {
            line.strip()
            for line in f
            if line.strip() and line.strip() != "StandardName"
        }

    logging.info(f"Loaded {len(fields)} RESO standard fields from '{file_path}'.")
    return fields


# ---------------------------------------------------------------------------
# 2. Key Unwrapping & Flattening Utilities
# ---------------------------------------------------------------------------
def unwrap_wrapper_keys(obj: Any) -> Any:
    """Recursively removes outer redundant wrapper keys like properties, data, result, etc."""
    wrapper_keys = {"properties", "data", "result", "json", "response", "output"}
    if isinstance(obj, dict) and len(obj) == 1:
        key = list(obj.keys())[0]
        if str(key).lower() in wrapper_keys and isinstance(obj[key], dict):
            return unwrap_wrapper_keys(obj[key])
    return obj


def to_pascal_case(key: str) -> str:
    """Converts camelCase keys to PascalCase (e.g., propertyType -> PropertyType)."""
    if not key:
        return key
    return key[0].upper() + key[1:]


def extract_flattened_keys(obj: Any) -> Set[str]:
    """Recursively flattens all JSON keys from nested dictionaries and lists."""
    keys = set()
    obj = unwrap_wrapper_keys(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            if isinstance(v, (dict, list)):
                keys.update(extract_flattened_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                keys.update(extract_flattened_keys(item))

    return keys


# ---------------------------------------------------------------------------
# 3. Path Precision Computation
# ---------------------------------------------------------------------------
def calculate_pp_from_cleaned_json(
    cleaned_json_val: Any,
    parse_status_val: Any,
    reso_allowed_fields: Set[str]
) -> Dict[str, Any]:
    """
    Computes 3 variants of Path Precision metrics and validates JSON status:
    1. pp_strict: Exact case match against RESO whitelist.
    2. pp_strict_normalized: Converted camelCase to PascalCase match.
    3. pp_soft: Case-insensitive match.
    """
    # 1. Check post-processing status
    status_str = str(parse_status_val).strip().upper() if pd.notna(parse_status_val) else ""
    if status_str == "FAILED/EMPTY":
        return {
            "is_json_valid": 0,
            "pp_strict": 0.0,
            "pp_strict_normalized": 0.0,
            "pp_soft": 0.0
        }

    # 2. Check empty JSON string
    if not isinstance(cleaned_json_val, str) or not cleaned_json_val.strip() or cleaned_json_val.strip() == "{}":
        return {
            "is_json_valid": 0,
            "pp_strict": 0.0,
            "pp_strict_normalized": 0.0,
            "pp_soft": 0.0
        }

    # 3. Parse JSON object
    try:
        json_obj = json.loads(cleaned_json_val)
    except Exception:
        return {
            "is_json_valid": 0,
            "pp_strict": 0.0,
            "pp_strict_normalized": 0.0,
            "pp_soft": 0.0
        }

    f_gen = extract_flattened_keys(json_obj)

    if not f_gen:
        return {
            "is_json_valid": 0,
            "pp_strict": 0.0,
            "pp_strict_normalized": 0.0,
            "pp_soft": 0.0
        }

    # Variant 1: Raw Strict Path Precision (Exact Case)
    intersection_strict = f_gen.intersection(reso_allowed_fields)
    pp_strict = len(intersection_strict) / len(f_gen)

    # Variant 2: Normalized Strict Path Precision (camelCase -> PascalCase)
    f_gen_pascal = {to_pascal_case(k) for k in f_gen}
    intersection_pascal = f_gen_pascal.intersection(reso_allowed_fields)
    pp_strict_normalized = len(intersection_pascal) / len(f_gen_pascal)

    # Variant 3: Soft Path Precision (Case Insensitive)
    f_gen_lower = {k.lower() for k in f_gen}
    reso_allowed_lower = {f.lower() for f in reso_allowed_fields}
    intersection_soft = f_gen_lower.intersection(reso_allowed_lower)
    pp_soft = len(intersection_soft) / len(f_gen_lower)

    return {
        "is_json_valid": 1,
        "pp_strict": round(pp_strict, 4),
        "pp_strict_normalized": round(pp_strict_normalized, 4),
        "pp_soft": round(pp_soft, 4)
    }


# ---------------------------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------------------------
def run_pp_evaluation(
    input_file: str,
    whitelist_file: str,
    output_detail_file: str,
    output_summary_file: str
) -> None:
    """Runs full Path Precision evaluation pipeline across all models."""
    reso_fields = load_reso_standard_fields(whitelist_file)

    if not os.path.exists(input_file):
        logging.error(f"Input file not found: '{input_file}'")
        return

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    logging.info(f"Loaded '{input_file}' containing {len(df)} records.")

    # Validation check: Path Precision requires postprocess.py output
    if "cleaned_json" not in df.columns or "parse_status" not in df.columns:
        logging.warning(
            "⚠️ CRITICAL WARNING: Columns 'cleaned_json' and/or 'parse_status' missing in input file! "
            "Path Precision strictly requires preprocessed datasets from 'postprocess.py' "
            "to flatten nested keys and remove raw markdown syntax noise."
        )

    logging.info("Calculating Path Precision metrics from 'cleaned_json' and 'parse_status'...")

    eval_results = df.apply(
        lambda row: calculate_pp_from_cleaned_json(
            row.get("cleaned_json", ""),
            row.get("parse_status", ""),
            reso_fields
        ),
        axis=1
    )

    eval_df = pd.DataFrame(list(eval_results))

    for col in eval_df.columns:
        df[col] = eval_df[col]

    # Model Summary Aggregation
    summary = df.groupby("model").agg(
        total_samples=("is_json_valid", "count"),
        json_pass_rate=("is_json_valid", lambda x: round((x.sum() / x.count()) * 100, 2)),
        mean_pp_strict=("pp_strict", lambda x: round(x.mean() * 100, 2)),
        mean_pp_strict_norm=("pp_strict_normalized", lambda x: round(x.mean() * 100, 2)),
        mean_pp_soft=("pp_soft", lambda x: round(x.mean() * 100, 2))
    ).reset_index()

    summary.columns = [
        "LLM Model",
        "Total Sample",
        "JSON Pass Rate (%)",
        "Raw Strict PP (%)",
        "Normalized Strict PP (%)",
        "Soft Path Precision (%)"
    ]

    logging.info("\n" + "=" * 80)
    logging.info(" PATH PRECISION BENCHMARK SUMMARY BY MODEL ")
    logging.info("=" * 80)
    print(summary.to_string(index=False))
    logging.info("=" * 80)

    # Save Output CSVs
    try:
        summary.to_csv(output_summary_file, index=False, encoding="utf-8-sig")
        logging.info(f"Path Precision summary report saved to: '{output_summary_file}'")
    except Exception as e:
        logging.error(f"Failed to save summary file '{output_summary_file}': {e}")

    try:
        df.to_csv(output_detail_file, index=False, encoding="utf-8-sig")
        logging.info(f"Detailed Path Precision results saved to: '{output_detail_file}'")
    except Exception as e:
        logging.error(f"Failed to save detail file '{output_detail_file}': {e}")


# ---------------------------------------------------------------------------
# 5. CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Path Precision Evaluation Pipeline (Requires postprocessed dataset from postprocess.py)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_cleaned_unified.csv",
        help="Path to post-processed CSV file from postprocess.py (must contain 'cleaned_json' & 'parse_status')"
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
        default="benchmark_cleaned_unified_with_pp.csv",
        help="Path to save row-level detailed Path Precision CSV"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="benchmark_pp_summary.csv",
        help="Path to save aggregated Path Precision summary report CSV"
    )

    args = parser.parse_args()

    run_pp_evaluation(
        input_file=args.input,
        whitelist_file=args.whitelist,
        output_detail_file=args.output_detail,
        output_summary_file=args.output_summary
    )


if __name__ == "__main__":
    main()