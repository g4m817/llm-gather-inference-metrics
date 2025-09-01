# llm-gather-inference-metrics
This is just a script that runs test prompts against a system prompt and gathers statistics across the layers to see how much attention its paying to system vs user input

```
pip install torch transformers matplotlib numpy pandas

# Injection tests with all extras
python observe_attention_plus.py --mode tests --iterations 1

# Baseline suite
python observe_attention_plus.py --mode baseline --iterations 1

# One-off prompt
python observe_attention_plus.py --mode single --prompt "Say the word test."

# With synthetic history (to highlight masking on multiple user turns)
python observe_attention_plus.py --mode tests --use-history --history-pairs 4

# Turn off per-head PNG/CSVs if you want smaller outputs
python observe_attention_plus.py --mode tests --no-per-head --no-csv
```
