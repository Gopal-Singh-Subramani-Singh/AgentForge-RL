from llm.model import LocalLLM


class DeveloperAgent:

    SYSTEM_PROMPT = """
You are the Developer Agent in a software engineering team.
Write correct Python code that satisfies all requirements.

When writing code:
- Return ONLY Python source code.
- Do not use markdown fences.
- Do not include explanations.
"""

    DEBUG_PROMPT = """
You are an expert Python debugger.

Your job is NOT to write code.

Analyze the implementation and pytest failures carefully.
Trace the algorithm using the failed examples.

Explain:
1. What is wrong.
2. Why the observed values differ from the expected values.
3. The exact logical change required.

Be concise and specific.
"""

    def __init__(self, llm):
        self.llm = llm

    def implement(self, task):
        return self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
Implement this task:

{task}

Return only the complete Python source file.
""",
            max_new_tokens=1024,
        )

    def analyze_failure(
        self,
        task,
        current_code,
        test_output,
    ):
        return self.llm.generate(
            system_prompt=self.DEBUG_PROMPT,
            user_prompt=f"""
ORIGINAL REQUIREMENTS:

{task}

CURRENT IMPLEMENTATION:

{current_code}

PYTEST RESULTS:

{test_output}

Diagnose the logical error.
Do NOT provide replacement source code.
""",
            max_new_tokens=512,
            do_sample=False,
        )

    def fix(
    self,
    task,
    current_code,
    test_output,
    diagnosis,
):
    prompt = f"""
You are repairing a Python implementation.

ORIGINAL REQUIREMENTS:

{task}

CURRENT IMPLEMENTATION:

{current_code}

PYTEST FAILURE:

{test_output}

TESTER DIAGNOSIS:

Problem:
{diagnosis["problem"]}

Location:
{diagnosis["location"]}

Cause:
{diagnosis["cause"]}

REQUIRED CHANGE:
{diagnosis["required_change"]}

Modify the implementation according to the Tester's required change.

IMPORTANT:

- The current implementation is known to be incorrect.
- You MUST make the requested change.
- Do not return the unchanged implementation.
- Preserve behavior that is already correct.
- Return ONLY the complete corrected Python source file.
"""

    return self.llm.generate(
        system_prompt=self.SYSTEM_PROMPT,
        user_prompt=prompt,
        max_new_tokens=1024,
        do_sample=False,
    )