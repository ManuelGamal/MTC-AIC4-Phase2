# Training Details

Our solution relies purely on the robust pre-trained representations of the `uetrack_base` checkpoint. We focused entirely on algorithmic inference-time optimizations (Dynamic Search Expansion, Modified Hanning Penalty, Native Tensor Duplication, and Optimized Box EMA) rather than fine-tuning the weights on the competition dataset.

Therefore, no custom training scripts are required to reproduce our solution.
