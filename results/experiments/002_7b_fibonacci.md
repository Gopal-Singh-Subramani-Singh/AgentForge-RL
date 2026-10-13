# Experiment 002 - Qwen2.5-Coder-7B Fibonacci

Model:
Qwen2.5-Coder-7B-Instruct

Quantization:
4-bit NF4

Hardware:
NVIDIA A100 80GB PCIe MIG 1g.10gb

Usable VRAM:
9.50 GB

Allocated VRAM:
5.18 GB

Reserved VRAM:
5.33 GB

Generation time:
5.64 seconds

Tests:
6/6 passed

Developer attempts:
1

Result:
SUCCESS

Observation:
The 7B 4-bit model solved the benchmark correctly on its first attempt.
The previous 3B FP16 model passed only 4/6 tests and failed to repair
its implementation after repeated feedback.
