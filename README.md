# When Benchmarks Measure the Pipeline: Instrument Validity in LLM Evaluation

**Anonymous Code and Data Repository for Peer Review**  
*AAAI-27 Workshop on Effective and Efficient Evaluation for AI Scientists (EvalAIScientists 2027)*

---

## Overview

This repository provides the complete, self-contained replication pipeline and evaluation data for the paper:  
**"When Benchmarks Measure the Pipeline: Instrument Validity in LLM Evaluation"**.

Using multi-step mathematical reasoning on GSM8K as a diagnostic testbed, this work demonstrates how subtle choices in evaluation harnesses—such as padding conventions, prompt positioning, answer extraction hierarchies, and unadjusted covariates—can distort measured reasoning capabilities by double-digit margins.

---

## Repository Structure

```
├── README.md                 <- Reproduction instructions and documentation
├── requirements.txt          <- Required Python libraries
├── reproduce_all.py          <- Master one-command reproduction script
├── src/
│   ├── answer_extractor.py   <- Prioritized 3-tier regex extraction hierarchy
│   ├── prompt_builder.py     <- Flawed Front vs. Symmetric Prompt builders
│   ├── model_inference.py    <- Greedy decoding loop with FP4 quantization and padding control
│   └── statistical_analysis.py <- Complete-case filtering, McNemar tests, GLMM regressions
└── data/
    ├── position_ablation_qwen.csv    <- Phase 3.5 Qwen evaluation outputs (375 runs)
    ├── position_ablation_llama.csv   <- Phase 3.5 Llama evaluation outputs (375 runs)
    ├── pilot_results_clean_qwen.csv  <- Phase 2 Qwen append-at-back dataset (for GLMM)
    ├── pilot_results_clean_llama.csv <- Phase 2 Llama append-at-back dataset (for GLMM)
    ├── signature_vectors.csv         <- Phase 1 10-dimensional signature vectors
    └── graph_statistics.csv          <- Phase 1 computational graph complexity statistics
```

---

## Quickstart: Replicate All Empirical Findings (No GPU Required)

To verify all tables, statistical tests, and findings reported in the paper from the saved empirical outputs:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master reproduction script
python reproduce_all.py
```

### Expected Output Summary:
1. **Table 1 (Phase 3.5 Symmetric Prompt)**:
   - **Qwen2.5-7B** ($N=64$): Str.Back = $90.6\%$, Str.Front = $82.8\%$, Echo.B = $0.0\%$, Echo.F = $1.6\%$, Lex.Back = $96.9\%$
   - **Llama-3.1-8B** ($N=59$): Str.Back = $83.1\%$, Str.Front = $74.6\%$, Echo.B = $0.0\%$, Echo.F = $1.7\%$, Lex.Back = $89.8\%$
   - **Mistral-7B** ($N=39$): Str.Back = $43.6\%$, Str.Front = $33.3\%$, Echo.B = $5.1\%$, Echo.F = $25.6\%$, Lex.Back = $76.9\%$
2. **McNemar's Exact Tests**:
   - Qwen2.5-7B: $p = 0.125$ (non-significant positional asymmetry)
   - Llama-3.1-8B: $p = 0.180$ (2 Front-only vs. 7 Back-only discordant pairs, non-significant)
   - Mistral-7B: $p = 0.008$ (persistent Front-echo gap of $25.6\%$ vs. $5.1\%$)
3. **Matched-Seed Cohort Sensitivity ($n=33$)**:
   - Flawed Front Prompt: $18.2\%$ retention, $54.5\%$ echo rate
   - Symmetric Front Prompt: $90.9\%$ retention, $3.0\%$ echo rate
4. **Table 2 (Clustered Logistic Regression, Finding 4)**:
   - **Qwen2.5-7B** ($N=195$): $\hat{\beta}_1 = 0.41$, $\mathrm{OR} = 1.50$, $95\%\ \mathrm{CI} = [-0.55, +1.36]$, $p = 0.405$
   - **Llama-3.1-8B** ($N=177$): $\hat{\beta}_1 = 1.09$, $\mathrm{OR} = 2.97$, $95\%\ \mathrm{CI} = [+0.36, +1.82]$, $p = 0.004$
   - Unadjusted Llama model (omitting $d_{\mathrm{lexical}}$): $\hat{\beta}_1 = 0.37$, $p = 0.17$ (specification-sensitive)

---

## Running Inference from Scratch (Requires GPU)

To execute live model inference across open-weight checkpoints:

```python
from src.model_inference import load_model_and_tokenizer, generate_response
from src.prompt_builder import build_symmetric_prompt

# 1. Load model with 4-bit quantization and left-padding
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct", padding_side="left")

# 2. Build Symmetric Prompt
prompt = build_symmetric_prompt(
    problem_text="There are 40 pupils in a class. Half like blue. One-fourth like green. How many like yellow?",
    rule="Suppose 5 more are added to the final amount.",
    position="front"
)

# 3. Generate response with greedy decoding (temperature=0.0)
output = generate_response(model, tokenizer, prompt, max_new_tokens=1024)
print(output)
```

---

## 3-Tier Answer Extraction Hierarchy

As described in Section 3.4 of the manuscript, output answers are extracted via:
1. **Tier 1 (`regex_hash`)**: `r"#{4}\s*\[?([+-]?\d+)\]?"` capturing canonical GSM8K delimiters with bracket tolerance.
2. **Tier 2 (`regex_natural`)**: Captures conclusion phrases such as *"the answer is"* or *"therefore the total is"*.
3. **Tier 3 (Fallback)**: Terminal integer token in the generation trace.

```python
from src.answer_extractor import extract_answer_with_tier

text = "The calculated amount is 15. #### [15]"
ans, tier = extract_answer_with_tier(text)
print(f"Extracted: {ans} via Tier {tier}")  # Extracted: 15 via Tier 1
```

---

## Ethical Statement & License
All evaluations use public open-weight language models on the public GSM8K benchmark. No human subjects, private user data, or proprietary web APIs were involved. Code is released under the MIT License for academic and research purposes.
