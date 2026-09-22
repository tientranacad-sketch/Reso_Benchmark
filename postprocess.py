import argparse
import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

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
# 1. JSON Repair & Recovery Utilities
# ---------------------------------------------------------------------------
def auto_close_json(json_str: str) -> str:
    """Automatically close truncated brackets and quotes caused by reaching max_tokens."""
    stack = []
    in_string = False
    escape = False

    for char in json_str:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack:
                    stack.pop()

    if in_string:
        json_str += '"'

    while stack:
        opening = stack.pop()
        if opening == '{':
            json_str += '}'
        elif opening == '[':
            json_str += ']'

    return json_str


def extract_first_json_block(text: str) -> str:
    """Extract the first valid JSON block using balanced brace counting."""
    start_idx = text.find('{')
    if start_idx == -1:
        return text

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i + 1]

    return text[start_idx:]


def fallback_regex_parser(text: str) -> Dict[str, Any]:
    """Extract independent key-value pairs using Regex when structural JSON parsing fails completely."""
    result = {}
    pattern = r'[\'"]([^\'"]+)[\'"]\s*:\s*([\'"][^\'"]*[\'"]|\d+\.?\d*|true|false|null|True|False|None)'
    matches = re.findall(pattern, text)
    for k, v in matches:
        v_clean = v.strip()
        if v_clean in ['true', 'True']:
            val = True
        elif v_clean in ['false', 'False']:
            val = False
        elif v_clean in ['null', 'None']:
            val = None
        else:
            try:
                val = json.loads(v_clean)
            except Exception:
                val = v_clean.strip('\'"')
        result[k] = val
    return result


# ---------------------------------------------------------------------------
# 2. Multi-stage Sanitizer Core
# ---------------------------------------------------------------------------
def clean_and_parse_json(raw_output: Any) -> Dict[str, Any]:
    """
    Multi-stage Sanitizer for LLM outputs:
    1. Extract from Markdown Code Block or balanced braces.
    2. Fix repeated key loops (e.g., typeOfPropertySubsubsub...).
    3. Try standard json.loads.
    4. Clean syntax errors (double quotes, trailing commas, Python booleans/None).
    5. Evaluate with ast.literal_eval for single-quoted strings.
    6. Recover truncated JSON strings (auto_close_json).
    7. Regex fallback extraction.
    """
    if pd.isna(raw_output):
        return {}

    cleaned = str(raw_output).strip()
    if not cleaned or cleaned.lower() == "nan":
        return {}

    # Stage 1: Extract from Markdown Code Block or balanced braces
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()
    else:
        cleaned = extract_first_json_block(cleaned)

    # Stage 2: Fix infinite key repetition loops
    cleaned = re.sub(r'("(?:Sub)*subcategory")', r'"typeOfPropertyCategory"', cleaned, flags=re.IGNORECASE)

    # Stage 3: Attempt standard JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Stage 4: Fix common LLM syntax anomalies
    cleaned_fix = cleaned.replace('""', '"')
    cleaned_fix = re.sub(r',+\s*([\}\]])', r'\1', cleaned_fix)  # Trailing commas
    cleaned_fix = re.sub(r'\bTrue\b', 'true', cleaned_fix)
    cleaned_fix = re.sub(r'\bFalse\b', 'false', cleaned_fix)
    cleaned_fix = re.sub(r'\bNone\b', 'null', cleaned_fix)

    try:
        parsed = json.loads(cleaned_fix)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Stage 5: Evaluate via ast.literal_eval (Handles single-quoted dictionaries)
    try:
        literal_dict = ast.literal_eval(cleaned)
        if isinstance(literal_dict, dict):
            return literal_dict
    except Exception:
        pass

    # Stage 6: Recover truncated JSON from max_tokens exhaustion
    try:
        repaired_str = auto_close_json(cleaned_fix)
        parsed = json.loads(repaired_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Stage 7: Regex fallback
    return fallback_regex_parser(cleaned)


# ---------------------------------------------------------------------------
# 3. Processing & Merging Pipeline
# ---------------------------------------------------------------------------
def process_and_merge_benchmarks(
    files_to_process: List[str],
    output_csv: str,
    deduplicate: bool = True
) -> None:
    """Post-process, sanitize, deduplicate, and merge benchmark result CSV files."""
    processed_dfs = []

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping...")
            continue

        logging.info(f"Post-processing data from: {file_path}...")
        
        df = pd.read_csv(
            file_path,
            usecols=lambda c: c in ['post_id', 'model', 'generated_json', 'raw_output', 'response', 'raw_json', 'original_text'],
            low_memory=False
        )

        if 'post_id' not in df.columns or 'model' not in df.columns:
            logging.warning(f"File {file_path} missing required 'post_id' or 'model' columns. Skipping...")
            continue

        df['post_id'] = pd.to_numeric(df['post_id'], errors='coerce')
        df = df.dropna(subset=['post_id', 'model'])
        df['post_id'] = df['post_id'].astype(int)

        json_col = None
        for col in ["generated_json", "raw_output", "response", "raw_json"]:
            if col in df.columns:
                json_col = col
                break

        if not json_col:
            logging.warning(f"No JSON output column found in file {file_path}.")
            continue

        cleaned_jsons = []
        parse_statuses = []
        extracted_keys = []
        keys_counts = []

        for raw_val in df[json_col]:
            parsed_dict = clean_and_parse_json(raw_val)
            is_valid = len(parsed_dict) > 0

            cleaned_jsons.append(json.dumps(parsed_dict, ensure_ascii=False))
            parse_statuses.append("SUCCESS" if is_valid else "FAILED/EMPTY")
            extracted_keys.append(list(parsed_dict.keys()))
            keys_counts.append(len(parsed_dict))

        df["source_file"] = os.path.basename(file_path)
        df["cleaned_json"] = cleaned_jsons
        df["parse_status"] = parse_statuses
        df["extracted_keys"] = extracted_keys
        df["extracted_keys_count"] = keys_counts

        if deduplicate:
            df['sort_order'] = df['parse_status'].apply(lambda x: 1 if x == "SUCCESS" else 0)
            df = df.sort_values(by=['post_id', 'model', 'sort_order'], ascending=[True, True, True])
            df = df.drop_duplicates(subset=['post_id', 'model'], keep='last').drop(columns=['sort_order'])

        processed_dfs.append(df)

    if not processed_dfs:
        logging.error("No valid benchmark data processed.")
        return

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    priority_cols = ["source_file", "post_id", "model", "parse_status", "extracted_keys_count", "cleaned_json", "extracted_keys"]
    remaining_cols = [c for c in combined_df.columns if c not in priority_cols]
    final_cols = [c for c in priority_cols + remaining_cols if c in combined_df.columns]

    combined_df = combined_df[final_cols]
    combined_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logging.info("\n================ SUMMARY POST-PROCESSING ================")
    logging.info(f"- Total processed records: {len(combined_df)}")
    logging.info(f"- Successfully parsed (SUCCESS): {len(combined_df[combined_df['parse_status'] == 'SUCCESS'])}")
    logging.info(f"- Failed or empty (FAILED/EMPTY): {len(combined_df[combined_df['parse_status'] == 'FAILED/EMPTY'])}")
    logging.info("- Breakdown by source file:")
    print(combined_df.groupby(["source_file", "parse_status"]).size().to_string())
    logging.info(f"\n[DONE] Sanitized output saved to: {output_csv}")


# ---------------------------------------------------------------------------
# 4. CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-stage Data Post-Processor & Sanitizer for LLM Outputs")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["benchmark_openrouter_results.csv", "benchmark_savegate_results.csv"],
        help="List of benchmark CSV result files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_cleaned_unified.csv",
        help="Path to output sanitized CSV file"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication by (post_id, model)"
    )

    args = parser.parse_args()

    process_and_merge_benchmarks(
        files_to_process=args.inputs,
        output_csv=args.output,
        deduplicate=not args.no_dedup
    )


if __name__ == "__main__":
    main()