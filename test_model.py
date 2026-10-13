import torch

from llm.model import LocalLLM


print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))


llm = LocalLLM("cuda:0")

result = llm.generate(
    system_prompt=(
        "You are an expert Python software developer."
    ),
    user_prompt="""
Write a Python function:

def fibonacci(n):

It should return the nth Fibonacci number.

Return only the Python code.
""",
)

print("\n--- RESPONSE ---")
print(result["text"])

print("\n--- PERFORMANCE ---")
print("Time:", round(result["time"], 2), "seconds")
print("Tokens:", result["tokens"])

if result["time"] > 0:
    print(
        "Tokens/sec:",
        round(result["tokens"] / result["time"], 2),
    )
