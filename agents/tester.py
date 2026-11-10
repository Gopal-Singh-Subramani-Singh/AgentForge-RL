import json


class TesterAgent:

    SYSTEM_PROMPT = """
You are the Tester Agent in a software engineering team.

Your job is to analyze failing pytest results and identify the
exact cause of the failure.

You do NOT write the full replacement program.

Return ONLY valid JSON using this format:

{
    "problem": "short description of the bug",
    "location": "function or area containing the bug",
    "cause": "why the current implementation fails",
    "required_change": "specific change the Developer must make"
}

Rules:

1. Base your diagnosis ONLY on:
   - the original project requirements,
   - the current implementation,
   - and the pytest output.

2. Compare expected values with actual values.

3. Inspect the current implementation carefully.

4. Give one concrete required change.

5. NEVER invent classes, functions, APIs, files, or architecture
   that are not explicitly required by the project requirements
   or demonstrated by the tests.

6. If pytest reports that a required module-level function is
   missing, the required change must be to add that exact
   module-level function.

7. Do not replace a required module-level API with a class-based
   API unless the original requirements explicitly require a class.

8. The diagnosis must directly explain the actual pytest failure.

9. Do not return Python code.

10. Do not use markdown.
"""

    def __init__(self, llm):
        self.llm = llm

    def diagnose(
        self,
        task,
        current_code,
        test_output,
    ):

        result = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
PROJECT REQUIREMENTS:

{task}

CURRENT IMPLEMENTATION:

{current_code}

PYTEST OUTPUT:

{test_output}

Identify the exact bug and required change.
""",
            max_new_tokens=512,
            do_sample=False,
        )

        text = result["text"].strip()
        if text.startswith("```json"):
            text = text[len("```json"):]
        
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()

        try:
            diagnosis = json.loads(text)

        except json.JSONDecodeError:
            diagnosis = {
                "problem": "Tester returned unstructured output",
                "location": "unknown",
                "cause": text,
                "required_change": text,
            }

        result["diagnosis"] = diagnosis

        return result
