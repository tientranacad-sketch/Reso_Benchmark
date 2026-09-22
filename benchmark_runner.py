import argparse
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
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

# Thread lock for safe concurrent CSV writing
csv_lock = threading.Lock()

# Provider configurations and supported LLM models
PROVIDER_CONFIGS = {
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_output": "benchmark_openrouter_results.csv",
        "models": [
            "meta-llama/llama-3.2-3b-instruct",
            "qwen/qwen3-32b",
            "meta-llama/llama-3.1-70b-instruct",
            "z-ai/glm-4.7-flash",
            "deepseek/deepseek-chat-v3.1",
            "x-ai/grok-4.5",
            "moonshotai/kimi-k3",
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ],
    },
    "savegate": {
        "env_key": "SAVEGATE_API_KEY",
        "base_url": "https://api.savegate.ai/v1",
        "default_output": "benchmark_savegate_results.csv",
        "models": [
            "gpt-5.4-mini",
            "gemini-2.5-flash",
            "claude-sonnet-4-6",
            "gpt-5.4",
        ],
    },
}

PROMPT_TEMPLATE = (
    "Standardize real estate data using reso.org 2.0;"
    "estimate geo long lat if address was provided;"
    "ignore fields listingId listingStatus description publicRemarks headline;"
    "use camelcase;"
    "output only minimized Flat JSON: {text_data}"
)


class InsufficientCreditsError(Exception):
    """Raised when the account runs out of credits or encounters HTTP 402."""
    pass


# ---------------------------------------------------------------------------
# 2. Helper Functions & API Engine
# ---------------------------------------------------------------------------
def check_openrouter_balance(api_key: str):
    """Check remaining credit balance on OpenRouter prior to batch execution."""
    if not api_key:
        return
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        limit = data.get("limit")
        usage = data.get("usage")
        limit_remaining = data.get("limit_remaining")
        logging.info(f"OpenRouter Credit Status: usage={usage}, limit={limit}, remaining={limit_remaining}")
        if limit_remaining is not None and limit_remaining <= 0:
            logging.warning("⚠️  WARNING: limit_remaining <= 0 — Requests may fail with HTTP 402.")
    except Exception as e:
        logging.warning(f"Failed to check OpenRouter balance: {e}")


def get_api_client(provider: str) -> OpenAI:
    """Initialize OpenAI-compatible client for the specified provider."""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Invalid provider: '{provider}'.")

    api_key = os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider}'. Please set {config['env_key']} in .env file.")

    return OpenAI(api_key=api_key, base_url=config["base_url"])


def call_llm_api(
    client: OpenAI,
    model_name: str,
    text_data: str,
    temperature: float = 0.1,
    max_retries: int = 6
) -> str:
    """Robust LLM completion call with retry strategy and backoff."""
    prompt_content = PROMPT_TEMPLATE.format(text_data=text_data)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=temperature
            )

            if not response.choices:
                raise ValueError("Response choices array is empty.")

            choice = response.choices[0]
            content = choice.message.content

            if content is None:
                raise ValueError(f"Content is None (finish_reason={choice.finish_reason})")

            return content.strip()

        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credits" in err_str.lower():
                raise InsufficientCreditsError(f"Insufficient credits for model {model_name}: {err_str}") from e

            # Network connection or parsing errors -> wait longer
            if any(k in err_str for k in ["Connection", "Expecting value", "NoneType", "502", "503", "504"]):
                sleep_time = 15
            else:
                sleep_time = 5 * (attempt + 1)

            if attempt < max_retries - 1:
                logging.warning(f"  [Retry {attempt + 1}/{max_retries}] {model_name} error -> Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                return f"ERROR: Connection or Server failure after {max_retries} retries. Details: {err_str}"


def process_task(client: OpenAI, post_id: int, provider: str, model_name: str, text_data: str, temperature: float) -> dict:
    """Task worker wrapper for ThreadPoolExecutor."""
    try:
        output_json = call_llm_api(client, model_name, text_data, temperature=temperature)
        return {
            "post_id": post_id,
            "provider": provider,
            "model": model_name,
            "original_text": text_data,
            "generated_json": output_json
        }
    except InsufficientCreditsError:
        return {"CRITICAL_ERROR": "INSUFFICIENT_CREDITS"}
    except Exception as e:
        return {
            "post_id": post_id,
            "provider": provider,
            "model": model_name,
            "original_text": text_data,
            "generated_json": f"ERROR: {str(e)}"
        }


# ---------------------------------------------------------------------------
# 3. Multithreaded Benchmark Engine
# ---------------------------------------------------------------------------
def run_benchmark_for_provider(
    provider: str,
    df: pd.DataFrame,
    output_csv: str,
    sample_limit: int,
    workers: int,
    temperature: float
):
    """Execute evaluation pipeline for a target provider using multithreading."""
    config = PROVIDER_CONFIGS[provider]
    models = config["models"]

    if provider == "openrouter":
        check_openrouter_balance(os.getenv(config["env_key"]))

    client = get_api_client(provider)

    # Smart Resume logic: Extract only successful (non-ERROR, non-NaN) executions
    successful_tasks = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(
                output_csv,
                encoding="utf-8-sig",
                usecols=lambda c: c in ["post_id", "model", "generated_json"],
                low_memory=False
            )

            df_success = df_existing[
                df_existing["generated_json"].notna() &
                ~df_existing["generated_json"].astype(str).str.strip().str.startswith("ERROR")
            ].copy()

            for _, r in df_success.iterrows():
                try:
                    pid = int(float(r["post_id"]))
                    successful_tasks.add((pid, str(r["model"]).strip()))
                except (ValueError, TypeError):
                    continue

            logging.info(f"[{provider}] Smart Resume: Recognized {len(successful_tasks)} successful executions from existing CSV.")
        except Exception as e:
            logging.warning(f"[{provider}] Failed to parse existing output file for smart resume: {e}")

    # Build queue of pending tasks (Runs missing tasks & retries failed tasks)
    sub_df = df.head(sample_limit)
    tasks_to_run = []
    for index, row in sub_df.iterrows():
        text_data = str(row.get("original_text", row.get("original_post", ""))).strip()
        pid = int(row["post_id"]) if "post_id" in row and pd.notna(row["post_id"]) else int(index)
        
        for model in models:
            if (pid, model) not in successful_tasks:
                tasks_to_run.append((pid, model, text_data))

    total_tasks = len(tasks_to_run)
    logging.info(f"=== STARTING MULTITHREADED BENCHMARK: {provider.upper()} ===")
    logging.info(f"Workers: {workers} | Pending Tasks to execute: {total_tasks} API calls\n")

    if total_tasks == 0:
        logging.info(f"[{provider}] All tasks are already completed successfully!\n")
        return

    # Multithreaded Execution Loop
    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(process_task, client, t[0], provider, t[1], t[2], temperature): t
            for t in tasks_to_run
        }

        for future in as_completed(future_to_task):
            completed_count += 1
            result = future.result()

            if "CRITICAL_ERROR" in result:
                logging.error("\n[!] CRITICAL ERROR: Insufficient credits. Halting execution.")
                os._exit(1)

            # Thread-safe write to CSV file
            with csv_lock:
                df_batch = pd.DataFrame([result])
                df_batch.to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding="utf-8-sig"
                )

            status = "❌" if str(result["generated_json"]).startswith("ERROR") else "✅"
            logging.info(
                f"[{completed_count}/{total_tasks}] {status} [{provider.upper()}] "
                f"Post #{result['post_id']} - Model: {result['model']}"
            )

    logging.info(f"Completed benchmark for provider {provider}! Saved to: {output_csv}\n")


# ---------------------------------------------------------------------------
# 4. Entrypoint (CLI Parser)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM Real Estate Standardization Benchmark Pipeline")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "savegate", "all"],
        default="all",
        help="Target LLM API provider to evaluate (default: all)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset.csv",
        help="Path to the input CSV dataset (default: dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (defaults to provider configuration if omitted)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of dataset samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent worker threads (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model sampling temperature (default: 0.1)"
    )

    args = parser.parse_args()

    # Input dataset validation
    input_file = args.input if os.path.exists(args.input) else "input_data.csv"
    if not os.path.exists(input_file):
        logging.error(f"Input file not found at '{args.input}' or 'input_data.csv'.")
        return

    df = pd.read_csv(input_file)

    providers_to_run = ["openrouter", "savegate"] if args.provider == "all" else [args.provider]

    for provider in providers_to_run:
        output_csv = args.output if args.output else PROVIDER_CONFIGS[provider]["default_output"]
        try:
            run_benchmark_for_provider(
                provider=provider,
                df=df,
                output_csv=output_csv,
                sample_limit=args.limit,
                workers=args.workers,
                temperature=args.temperature
            )
        except Exception as e:
            logging.error(f"Error executing benchmark for provider '{provider}': {e}")


if __name__ == "__main__":
    main()