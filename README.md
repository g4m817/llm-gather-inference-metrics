# llm-gather-inference-metrics
A zero-perturbation “observability” helper for LLM inference. This script runs a model as-is and, per prompt, captures rich telemetry from the first decode steps—per-layer/ per-head attention to system vs. user tokens, plus decoded tokens—then saves ready-to-inspect artifacts. For each run it emits:
- Global heatmaps: attn_share_system.png, attn_share_user.png, attn_share_diff_sys_minus_user.png
- Per-head grids per layer (PNGs) and optional CSVs (heads_system_share_layerXX.*)
- A compact tokens timeline image
- output.txt (generated text) and meta.txt (config, model/device, shapes)
- CSV dumps: system_share.csv, user_share.csv, diff_sys_minus_user.csv

It supports multi-turn masking of <|system|> / <|user|> / <|assistant|> spans, synthetic history injection, repeatable runs via seeds, and works with attn_implementation="eager" to expose attentions—no model surgery required.

I used this to find interesting metrics, that lead to [another PoC](https://github.com/g4m817/llm-detect-head-tripwire).

```
pip install torch transformers matplotlib numpy pandas

# Injection tests with all extras
python observe_attention_plus.py --mode tests --iterations 1

# Baseline suite
python observe_attention_plus.py --mode baseline --iterations 1

# One-off prompt
python observe_attention_plus.py --mode single --prompt "Say the word test."

# Turn off per-head PNG/CSVs if you want smaller outputs
python observe_attention_plus.py --mode tests --no-per-head --no-csv
```
