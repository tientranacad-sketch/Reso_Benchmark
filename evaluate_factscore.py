import argparse
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Logging & Environment Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

load_dotenv(override=True)

# LLM Judge prompt tailored for Vietnamese Real Estate domain verification
SYSTEM_PROMPT = """
Bạn là một Trọng tài Đánh giá Dữ liệu (LLM Judge) chuyên nghiệp trong lĩnh vực Bất động sản Việt Nam.
Nhiệm vụ của bạn là xác minh tính chính xác ngữ nghĩa của duy nhất một trường dữ liệu (Atomic Fact Verification)
được trích xuất từ văn bản tin rao bất động sản thô.

### QUY TẮC ĐÁNH GIÁ CHẶT CHẼ:
1. Độc lập Ngữ cảnh (Atomic Verification):
   - Bạn chỉ đánh giá trường dữ liệu được cung cấp so với văn cảnh (Context) bài đăng.
   - Không được tự suy luận bắc cầu phức tạp hoặc bịa đặt thông tin không có trong Context.

2. Quy đổi Thuật ngữ Bất động sản Việt Nam Chuẩn hóa:
   - "Tỏi" / "Tỷ" = 1,000,000,000 VNĐ (ví dụ: "8.5 tỏi" -> ListPrice: 8500000000).
   - "Củ" / "Mắm" / "Triệu" = 1,000,000 VNĐ (ví dụ: "50 củ" -> 50000000).
   - "Trệt", "Lầu": "1 trệt 2 lầu" -> Tổng số tầng/Stories = 3 tầng.
   - Trạng thái pháp lý: "Sổ hồng riêng", "SHR", "Sổ đỏ chính chủ" -> LegalStatus: "Sổ hồng/Sổ đỏ".
   - Hướng nhà: "Đông Nam", "Tây Bắc", "Nam"... -> Direction: Chuẩn hóa theo tiếng Việt.

3. Định dạng Đầu ra (Output Format):
   - Trả về JSON duy nhất chứa 2 trường:
     + "is_supported": boolean (true nếu giá trị hoàn toàn đúng hoặc phù hợp ngữ cảnh, false nếu sai hoặc ảo tưởng).
     + "reason": chuỗi giải thích ngắn gọn nguyên nhân.

### VÍ DỤ MINH HỌA ĐÁNH GIÁ (FEW-SHOT EXAMPLES):

Ví dụ 1:
[CONTEXT]: Bán nhà hẻm xe hơi đường Nguyễn Thị Minh Khai Quận 1, giá 12.5 tỏi, diện tích 60m2, 4 phòng ngủ, sổ hồng riêng.
[FIELD]: ListPrice = 12500000000
[OUTPUT]: {"is_supported": true, "reason": "12.5 tỏi quy đổi đúng thành 12,500,000,000 VNĐ"}

Ví dụ 2:
[CONTEXT]: Bán nhà hẻm xe hơi đường Nguyễn Thị Minh Khai Quận 1, giá 12.5 tỏi, diện tích 60m2, 4 phòng ngủ, sổ hồng riêng.
[FIELD]: ListPrice = 12500000
[OUTPUT]: {"is_supported": false, "reason": "Giá quy đổi sai nghiêm trọng (12.5 triệu thay vì 12.5 tỷ VNĐ)"}

Ví dụ 3:
[CONTEXT]: Cần bán căn hộ Chung cư Vinhomes Central Park 2PN 2WC full nội thất đẹp giá 4.2 tỷ.
[FIELD]: BedroomsTotal = 2
[OUTPUT]: {"is_supported": true, "reason": "2PN khớp với 2 phòng ngủ trong context"}

Ví dụ 4:
[CONTEXT]: Cần bán căn hộ Chung cư Vinhomes Central Park 2PN 2WC full nội thất đẹp giá 4.2 tỷ.
[FIELD]: BathroomsFull = 3
[OUTPUT]: {"is_supported": false, "reason": "Context ghi 2WC nhưng extracted value lại là 3 phòng tắm"}

Ví dụ 5:
[CONTEXT]: Đất thổ cư Củ Chi 100m2 (5x20m), đường nhựa 8m, pháp lý vi bằng công chứng.
[FIELD]: LegalStatus = "Sổ hồng riêng"
[OUTPUT]: {"is_supported": false, "reason": "Context ghi 'vi bằng' nhưng trích xuất thành 'Sổ hồng riêng'"}
"""


# ---------------------------------------------------------------------------
# 2. JSON Parsing & Flattening Helpers
# ---------------------------------------------------------------------------
def flatten_json(data: Any, parent_key: str = "") -> Dict[str, Any]:
    """Recursively flatten nested JSON dictionaries and arrays into key-value pairs."""
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}_{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(flatten_json(v, parent_key=new_key).items())
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_json(item, parent_key=f"{new_key}_{idx}").items())
                    else:
                        items.append((f"{new_key}_{idx}", item))
            else:
                items.append((new_key, v))
    return dict(items)


def clean_and_parse_json(raw_output: Any) -> Dict[str, Any]:
    """Extract and parse structured JSON dictionary from raw model generation string."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return {}
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return flatten_json(parsed)
        return {}
    except Exception:
        return {}


def load_reso_whitelist(file_path: str = "reso_fields.txt") -> Tuple[Set[str], Dict[str, str]]:
    """Load RESO standard field whitelist and construct lowercase lookup map."""
    fields = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            fields = {line.strip() for line in f if line.strip()}
    else:
        logging.warning(f"Whitelist file '{file_path}' not found. Using default RESO standard subset.")
        fields = {
            "listingAgentFullName", "listingAgentDirectPhone", "contactPhone",
            "propertyType", "propertySubType", "city", "district", "ward",
            "stateOrProvince", "country", "unparsedAddress", "listPrice",
            "currency", "livingArea", "lotSize", "bedrooms", "bathrooms",
            "yearBuilt", "parking", "mlsStatus", "remarks", "legalDescription",
            "lotFeatures", "latitude", "longitude"
        }
    lower_map = {f.lower(): f for f in fields}
    return fields, lower_map


# ---------------------------------------------------------------------------
# 3. LLM Judge Atomic Fact Verification
# ---------------------------------------------------------------------------
def verify_single_field_judge(
    client: OpenAI,
    context: str,
    field_name: str,
    field_value: Any,
    judge_model: str = "gpt-4o",
    max_retries: int = 3
) -> Optional[bool]:
    """Verify semantic accuracy of a single extracted field against raw text context."""
    user_prompt = f"""[CONTEXT TIN RAO BĐS]
{context}

[TRƯỜNG CẦN KIỂM TRA]
Field Name: {field_name}
Extracted Value: {field_value}

Hãy đưa ra phán quyết "is_supported" dưới dạng JSON."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                response_format={"type": "json_object"},
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result_str = response.choices[0].message.content
            result_json = json.loads(result_str)
            return bool(result_json.get("is_supported", False))
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to verify field '{field_name}': {e}")
                return None
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------------------------
def run_factscore_evaluation(
    csv_file_path: str,
    whitelist_file_path: str,
    sample_size_per_model: int,
    checkpoint_file: str,
    output_report_path: str,
    judge_model: str
):
    """Run end-to-end FActScore benchmark pipeline."""
    savegate_key = os.getenv("SAVEGATE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if savegate_key:
        api_key = savegate_key
        base_url = os.getenv("SAVEGATE_BASE_URL", "https://api.savegate.ai/v1")
    elif openai_key:
        api_key = openai_key
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    else:
        raise ValueError("Missing API key! Please set SAVEGATE_API_KEY or OPENAI_API_KEY in .env")
    client = OpenAI(api_key=api_key, base_url=base_url)
    _, lower_map = load_reso_whitelist(whitelist_file_path)

    df = pd.read_csv(csv_file_path, low_memory=False, on_bad_lines="skip")

    # 1. Filter out malformed model names
    df = df[
        df["model"].notna() 
        & (df["model"].astype(str).str.strip() != "") 
        & (~df["model"].astype(str).str.lower().isin(["none", "nan", "null", "model"]))
    ].copy()
    valid_models = df["model"].unique()

    if len(valid_models) == 0:
        logging.error("No valid models found in the input CSV file.")
        return

    # 2. Extract intersection of post_ids shared across ALL models
    post_ids_per_model = [
        set(df[df["model"] == m]["post_id"].dropna().unique())
        for m in valid_models
    ]
    common_post_ids = list(set.intersection(*post_ids_per_model))
    logging.info(f"Total common post_ids shared across all valid models: {len(common_post_ids)}")

    # 3. Fix random sample seed for fair global comparison
    if len(common_post_ids) >= sample_size_per_model:
        np.random.seed(42)
        global_sampled_post_ids = np.random.choice(common_post_ids, size=sample_size_per_model, replace=False)
    else:
        logging.warning(f"Common set has only {len(common_post_ids)} samples (requested: {sample_size_per_model}). Using all available.")
        global_sampled_post_ids = common_post_ids

    # Load persistent evaluation checkpoint cache
    progress_cache = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            progress_cache = json.load(f)

    summary_metrics = []

    for model_name in valid_models:
        logging.info(f"\n==================================================")
        logging.info(f"Evaluating Model: {model_name}")

        df_model_sample = df[(df["model"] == model_name) & (df["post_id"].isin(global_sampled_post_ids))].copy()
        record_scores = []

        for idx, row in df_model_sample.iterrows():
            post_id = str(row["post_id"])
            context = str(row["original_text"])
            raw_json_str = row["generated_json"]

            parsed_json = clean_and_parse_json(raw_json_str)
            if not isinstance(parsed_json, dict) or not parsed_json:
                record_scores.append(0.0)
                continue

            supported_count = 0
            evaluable_fields_count = 0

            for k, v in parsed_json.items():
                if v is None:
                    continue

                key_str = str(k).strip()
                key_lower = key_str.lower()

                # Filter out fields non-compliant with RESO whitelist
                if key_lower not in lower_map:
                    continue

                canonical_key = lower_map[key_lower]
                cache_key = f"{model_name}_{post_id}_{canonical_key}_{v}"

                if cache_key in progress_cache:
                    is_supported = progress_cache[cache_key]
                else:
                    is_supported = verify_single_field_judge(
                        client=client,
                        context=context,
                        field_name=canonical_key,
                        field_value=v,
                        judge_model=judge_model
                    )
                    if is_supported is None:
                        logging.error(f"HALTED: API failure at model '{model_name}', post_id '{post_id}'. Check API credits.")
                        return

                    progress_cache[cache_key] = is_supported
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(progress_cache, f, ensure_ascii=False, indent=2)

                if is_supported is not None:
                    evaluable_fields_count += 1
                    if is_supported is True:
                        supported_count += 1

            score = (supported_count / evaluable_fields_count) if evaluable_fields_count > 0 else 0.0
            record_scores.append(score)

        final_factscore = float(np.mean(record_scores)) * 100 if record_scores else 0.0
        logging.info(f"==> FACTSCORE ({model_name}): {final_factscore:.2f}%")

        summary_metrics.append({
            "model": model_name,
            "evaluated_samples": len(record_scores),
            "factscore_percent": round(final_factscore, 2)
        })

    df_report = pd.DataFrame(summary_metrics)
    df_report.to_csv(output_report_path, index=False, encoding="utf-8-sig")
    logging.info("\n================ SUMMARY FACTSCORE BENCHMARK ================")
    print(df_report.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. CLI Parser Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FActScore Evaluation Pipeline for RESO 2.0 Real Estate Benchmarks")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_openrouter_results.csv",
        help="Path to the benchmark results CSV file"
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default="reso_fields.txt",
        help="Path to the RESO whitelist text file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of intersection post_id samples to evaluate per model (default: 200)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="factscore_checkpoint.json",
        help="Path to progress cache JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="factscore_summary_report.csv",
        help="Path to output report CSV file"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="LLM Judge model name (default: gpt-4o)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input benchmark file not found: '{args.input}'")
        return

    run_factscore_evaluation(
        csv_file_path=args.input,
        whitelist_file_path=args.whitelist,
        sample_size_per_model=args.sample_size,
        checkpoint_file=args.checkpoint,
        output_report_path=args.output,
        judge_model=args.judge_model
    )


if __name__ == "__main__":
    main()