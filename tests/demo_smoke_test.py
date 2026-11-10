from pathlib import Path
import importlib

required_packages = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
    "gymnasium",
    "ray",
]

print("=== DEMO PRE-FLIGHT CHECK ===")

for package in required_packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"OK package: {package} ({version})")
    except Exception as exc:
        raise SystemExit(
            f"FAILED package: {package}: {exc}"
        )

import torch

CHECKPOINT = Path(
    "results/checkpoints/marl_ppo_alpha_0_25"
)

required_files = [
    Path("integration/rl_scheduler.py"),
    Path("integration/rl_bank_runner.py"),
    Path("integration/rl_team_runner.py"),
    Path("agents/developer.py"),
    Path("agents/tester.py"),
    Path("agents/documentation.py"),
    Path("agents/reviewer.py"),
    Path("llm/model_7b.py"),
]

for path in required_files:
    assert path.exists(), f"Missing: {path}"
    print("OK file:", path)

assert CHECKPOINT.exists(), (
    f"Missing PPO checkpoint: {CHECKPOINT}"
)

print("OK: PPO checkpoint")

assert torch.cuda.is_available(), (
    "CUDA is not available"
)

print(
    "OK: CUDA:",
    torch.cuda.get_device_name(0),
)

free_bytes, total_bytes = torch.cuda.mem_get_info()

print(
    "Free GPU GB:",
    round(free_bytes / 1024**3, 2),
)

print(
    "Total GPU GB:",
    round(total_bytes / 1024**3, 2),
)

print("\nPRE-FLIGHT CHECK PASSED")
