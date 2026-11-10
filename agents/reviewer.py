import json


class ReviewerAgent:

    SYSTEM_PROMPT = """
You are the Reviewer Agent in a software engineering team.

Your job is to inspect the finished project and determine whether
it satisfies the original requirements.

You will receive:
- Original project requirements
- Source files
- Test results
- Documentation

Evaluate:
1. Requirement coverage
2. Code completeness
3. Test status
4. Documentation accuracy
5. Obvious inconsistencies

Return ONLY valid JSON:

{
    "approved": true,
    "score": 0.95,
    "issues": [],
    "send_back_to": "none",
    "reason": "Short explanation"
}

If problems exist:

{
    "approved": false,
    "score": 0.70,
    "issues": [
        "Description of issue"
    ],
    "send_back_to": "developer",
    "reason": "Why the project should be revised"
}

send_back_to MUST be one of:

developer
tester
documentation
none

Rules:

- score must be between 0 and 1.
- Do not invent requirements.
- Do not reject a project for optional improvements.
- If all required functionality is implemented, tests pass,
  and documentation is accurate, approve it.
- Return JSON only.
- Do not use markdown fences.
"""

    def __init__(self, llm):
        self.llm = llm

    def review(
        self,
        task,
        project_files,
        test_output,
    ):
        files_text = ""

        for filename, contents in project_files.items():
            files_text += f"""

FILE: {filename}

{contents}

"""

        result = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
ORIGINAL REQUIREMENTS:

{task}

PROJECT FILES:

{files_text}

TEST RESULTS:

{test_output}

Review the project.
""",
            max_new_tokens=768,
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
            review = json.loads(text)

        except json.JSONDecodeError:
            review = {
                "approved": False,
                "score": 0.0,
                "issues": [
                    "Reviewer returned invalid JSON"
                ],
                "send_back_to": "none",
                "reason": text,
            }

        result["review"] = review

        return result
