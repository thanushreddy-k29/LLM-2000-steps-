# Plivo 2,000 Step LLM Speedrun Submission

This repository contains my final submission for the Plivo 2,000 Step LLM Speedrun challenge[cite: 1]. The goal of this project was to optimize a mediocre baseline GPT-style model under strict computational and architectural constraints[cite: 1, 6].

## Final Performance Metric
* **Bits Per Byte (BPB):** 1.9457
* **Parameter Count:** 1,871,232
* **Total Optimizer Steps:** 2,000

## Challenge Constraints
This model strictly adheres to the hard caps defined in the assignment[cite: 1, 6]:
* Maximum 2,000 optimizer steps[cite: 1].
* Maximum 2,000,000 total parameters[cite: 1].
* Trained exclusively on the provided `train_corpus.txt`[cite: 1].
* Built using pure PyTorch and standard libraries (no external pre-trained weights, transformers, or custom CUDA kernels)[cite: 1].
* Designed for CPU execution[cite: 1].

## Repository Structure
* `model.py` - The optimized GPT architecture (4 layers, 6 attention heads, 192 embedding dimension, tied weights)[cite: 3].
* `train.py` - The training loop, utilizing AdamW, a 150-step linear warmup, and cosine learning rate decay[cite: 6].
* `tokenizer.py` - A strict, byte-level fallback tokenizer capable of lossless UTF-8 encoding/decoding[cite: 5].
* `evaluate.py` - The official evaluation script used for scoring the model[cite: 2].
* `ckpt.pt` - The final saved model weights and step count[cite: 1, 4].
* `RUNLOG.md` - A step-by-step log of experimental iterations, hypotheses, and dev BPB results[cite: 1, 4].
* `NOTES.md` - A concise summary of the final configuration and why it succeeded[cite: 1, 4].
* `SUMMARY.html` - An AI-assisted technical breakdown of the architecture, methodology, and human-machine workload distribution[cite: 1, 4].

## Evaluation Instructions
To verify the Bits Per Byte (bpb) score on a hidden text file, run the following command unmodified from the root of this directory[cite: 1, 2]:

```bash
python evaluate.py --checkpoint ckpt.pt --text_file <any_text_file>
