# Real Estate Data Standardization Benchmark

This repository contains the benchmark execution and metric evaluation pipeline for standardizing Vietnamese real estate listings into RESO 2.0 flat JSON schema across multiple Large Language Models via OpenRouter and Savegate.

---

## Repository Structure
```text
.
├── benchmark_runner.py          # Multithreaded LLM extraction engine
├── postprocess.py               # Data post-processing & JSON sanitizer utility
├── evaluate_factscore.py        # FactScore evaluation script
├── evaluate_differentiationscore.py        # Differentiation Score evaluation script
├── evaluate_pathprecision.py    # Path Precision evaluation script
├── dataset.csv                  # Raw input dataset
├── reso_fields.txt              # RESO whitelist fields
├── requirements.txt             # Project dependencies
├── .gitignore                   # Excluded paths and secrets
└── README.md                    # Project documentation
```

## Environment Setup

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/tientranacad-sketch/Reso_Benchmark.git](https://github.com/tientranacad-sketch/Reso_Benchmark.git)
cd Reso_Benchmark

pip install -r requirements.txt
```
### 2. Configure API Keys
Create a .env file in the root directory:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
SAVEGATE_API_KEY=your_savegate_api_key_here
```

## Benchmark Execution (`benchmark_runner.py`)
The `benchmark_runner.py` script provides a multithreaded engine to run data extraction across target LLM models in parallel.

### **Usage Examples**
```bash
# Run OpenRouter models
python benchmark_runner.py --provider openrouter

# Run Savegate models
python benchmark_runner.py --provider savegate

# Run custom dataset (Smoke Test)
python benchmark_runner.py --input my_data.csv --limit 100 --workers 5

# Run full benchmark (All providers, 1,000 samples)
python benchmark_runner.py --provider all
```

### Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--provider` | `all` | Target provider to benchmark (`openrouter`, `savegate`, or `all`). |
| `--input` | `dataset.csv` | Path to the input CSV dataset. |
| `--output` | *Auto* | Custom CSV path for output results. |
| `--limit` | `1000` | Number of dataset samples to evaluate. |
| `--workers` | `5` | Number of concurrent worker threads for parallel requests. |
| `--temperature` | `0.1` | Model sampling temperature. |

## Data Post-Processing & Sanitization (`postprocess.py`)
The `postprocess.py` script cleans, repairs, and unifies raw LLM outputs. It resolves common LLM syntax issues (truncated JSON, markdown blocks, duplicate key loops, single quotes) and deduplicates entries to produce a clean dataset (`cleaned_json`, `parse_status`, `extracted_keys`) required for structural evaluation metrics (such as Differentiation Score and Path Precision).

### **Usage Examples**
```bash
# Process and merge default benchmark result files
python postprocess.py

# Process specific benchmark files and save to custom output
python postprocess.py \
  --inputs benchmark_openrouter_results.csv benchmark_savegate_results.csv \
  --output benchmark_cleaned_unified.csv
```
### **CLI Arguments**
|Argument|Default|Description|
|---|---|---|
|`--inputs`| `benchmark_openrouter_results.csv`,`benchmark_savegate_results.csv`|List of benchmark CSV result files to process.|
|`--output`|`benchmark_cleaned_unified.csv`|Path to save the sanitized CSV result.|
|`--no-dedup`|False|Flag to disable deduplication on `(post_id, model)`.|

## Metric Evaluation Suite
Each evaluation metric operates independently on the benchmark or sanitized datasets.

### 1. FactScore Evaluation (`evaluate_factscore.py`)
Calculates semantic accuracy (Factuality Score) via GPT-4o as an LLM Judge by performing atomic verification on extracted RESO fields against raw source text.

#### **Usage Examples**
```bash
# Evaluate OpenRouter benchmark results
python evaluate_factscore.py \
  --input benchmark_openrouter_results.csv \
  --output factscore_openrouter_summary.csv

# Evaluate Savegate benchmark results
python evaluate_factscore.py \
  --input benchmark_savegate_results.csv \
  --output factscore_savegate_summary.csv
```

#### **CLI Arguments**
|Argument|Default|Description|
|---|---|---|
|`--input`|`benchmark_openrouter_results.csv`|Path to the benchmark results CSV file.|
|`--whitelist`|`reso_fields.txt`|Path to the RESO field whitelist file.|
|`--sample-size`|200|Number of intersection post_id samples to evaluate per model.|
|`--checkpoint`|`factscore_checkpoint.json`|Path to JSON cache file for persistent checkpointing.|
|`--output`|`factscore_summary_report.csv`|Path to output summary report CSV file.|
|`--judge-model`|gpt-4o|LLM Judge model name used for verification.|

### 2. Differentiation Score Evaluation (`evaluate_differentiationscore.py`)
Calculates **Differentiation Score (DS)**, measuring schema structural deviation. It calculates $1.0 - \text{TF-IDF Cosine Similarity}$ between full generated JSON and whitelist-filtered JSON. Lower score is better ($0.0$ indicates perfect schema adherence).

> **⚠️ MANDATORY PREREQUISITE:**  
> Unlike `evaluate_factscore.py` (which can evaluate raw outputs), **`evaluate_differentiationscore.py` strictly requires the post-processed CSV dataset** generated by `postprocess.py` (e.g., `benchmark_cleaned_unified.csv`).  
> Running DS evaluation directly on raw LLM output containing Markdown blocks (````json ... ````), unparsed escape characters, or syntax errors will invalidate TF-IDF character vectorization and lead to incorrect/skewed DS scores.

#### **Usage Examples**
```bash
# Step 1: Ensure postprocess.py has been executed
python postprocess.py

# Step 2: Run Differentiation Score evaluation on sanitized dataset
python evaluate_differentiationscore.py

# Custom input/output execution
python evaluate_differentiationscore.py \
  --input benchmark_cleaned_unified.csv \
  --whitelist reso_fields.txt \
  --output-detail benchmark_cleaned_unified_with_ds.csv \
  --output-summary benchmark_ds_summary.csv
```

#### **CLI Arguments**
|Argument|Default|Description|
|---|---|---|
|`--input`|`benchmark_cleaned_unified.csv`|Path to sanitized CSV file from postprocess.py.|
|`--whitelist`|`reso_fields.txt`|Path to RESO field whitelist file.|
|`--output-detail`|`benchmark_cleaned_unified_with_ds.csv`|Output file for row-level DS scores.|
|`--output-summary`|`benchmark_ds_summary.csv`|Output file for aggregated model summary report.|

### Path Precision Evaluation (`evaluate_pathprecision.py`)
Calculates structural key precision across 3 strictness levels along with JSON pass rate:
* JSON Pass Rate (%): Percentage of valid, non-empty, parsable JSON outputs.
* Raw Strict PP (%): Exact case-sensitive key match ratio against RESO whitelist.
* Normalized Strict PP (%): Normalized camelCase to PascalCase key match ratio.
* Soft Path Precision (%): Case-insensitive key match ratio.

>⚠️ MANDATORY PREREQUISITE:
>`evaluate_pathprecision.py` strictly requires the post-processed CSV dataset generated by `postprocess.py` (containing `cleaned_json` and `parse_status`). Unparsed syntax noise or redundant wrapper keys (`properties`, `data`, `result`) in raw outputs will break key flattening and corrupt precision metrics.

#### **Usage Examples**
```bash
# Step 1: Run postprocessor
python postprocess.py

# Step 2: Evaluate Path Precision
python evaluate_pathprecision.py \
  --input benchmark_cleaned_unified.csv \
  --whitelist reso_fields.txt \
  --output-detail benchmark_cleaned_unified_with_pp.csv \
  --output-summary benchmark_pp_summary.csv
```

#### **CLI Arguments**
|Argument|Default|Description|
|---|---|---|
|`--input`|`benchmark_cleaned_unified.csv`|Mandatory: Path to post-processed CSV from postprocess.py.|
|`--whitelist`|`reso_fields.txt`|Path to RESO field whitelist file.|
|`--output-detail`|`benchmark_cleaned_unified_with_pp.csv`|Output file for row-level Path Precision metrics.|
|`--output-summary`|`benchmark_pp_summary.csv`|Output file for aggregated model summary report.|