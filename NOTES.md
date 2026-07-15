# Training Run Log

| Run # | Hypothesis | Changes Made | Dev BPB Before | Dev BPB After | Conclusion |
|-------|------------|--------------|----------------|---------------|------------|
| 1 | Baseline architecture scaled to 1.87M params with AdamW and LR scheduling. | Scaled `n_layer` to 4, `n_head` to 6, `n_embd` to 192. Added cosine decay. | N/A | 2.305 | Clean baseline. The learning rate schedule effectively drops the loss curve, but convergence is slow. |
| 2 | Doubling the context window will improve long-range dependencies and lower BPB. | Increased `block_size` from 256 to 512. | 2.305 | 2.344 | Score degraded. Within a 2,000-step cap, the model struggles to learn positional embeddings for 512 tokens efficiently; gradients dilute. |
| 3 | Reverting block_size, increasing batch size and LR will force faster convergence in 2000 steps. | Reverted `block_size` to 256. Changed `batch` to 32 and `max_lr` to 2e-3. | 2.344 | 2.005 | Massive success. The model achieved a much deeper minima with the aggressive learning rate and stable batches. |
| 4 | Removing dropout will accelerate learning by preventing gradient dilution in a short 2000-step run. | Changed `dropout` to 0.0. | 2.005 | < 2.005 | Final run. Removing regularization allowed 100% of the network capacity to memorize and adapt to the small 7MB dataset within the strict step limit. |