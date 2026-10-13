from llm.model_7b import LocalLLM7B

llm = LocalLLM7B("cuda:0")

result = llm.generate(
    system_prompt="""
You are an expert Python developer.
Return only valid Python code.
""",
    user_prompt="""
Implement fibonacci(n).

Requirements:
- fibonacci(0) == 0
- fibonacci(1) == 1
- fibonacci(2) == 1
- fibonacci(5) == 5
- fibonacci(10) == 55
- negative input raises ValueError
""",
)

print("\nOUTPUT:")
print(result["text"])

print("\nTime:", round(result["time"], 2))
print("Tokens:", result["tokens"])
