# Plivo 2,000 Step LLM Speedrun Submission

This repository contains my final submission for the Plivo 2,000 Step LLM Speedrun challenge. The goal of this project was to optimize a mediocre baseline GPT-style model under strict computational and architectural constraints.

## Final Performance Results

By optimizing the optimizer dynamics, implementing a learning rate scheduler, and removing dropout to maximize short-term adaptation within the short step window, the model achieved the following final metrics on the local evaluation set:

* **Final Training Loss:** 1.2404
* **Final Dev BPB (Bits Per Byte):** 1.9457

**Official Evaluator JSON Output:**
```json
{"bpb": 1.9457, "n_params": 1871232, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}
