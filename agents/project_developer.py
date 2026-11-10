import json


class ProjectDeveloperAgent:

    SYSTEM_PROMPT = """
You are the Developer Agent in a software engineering team.

You build complete Python applications.

You MUST return ONLY valid JSON.

Format:

{
    "files": {
        "filename.py": "complete file contents",
        "another_file.py": "complete file contents"
    }
}

Rules:

- Do not use markdown.
- Do not use code fences.
- Return valid JSON only.
- Include complete file contents.
- Follow every requirement.
- Do not include tests unless requested.
"""

    def __init__(self, llm):
        self.llm = llm

    def implement(self, task):

        result = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
Build the following project:

{task}

Return the complete project as JSON.
""",
            max_new_tokens=2048,
            do_sample=False,
        )

        result["project"] = self._parse(
            result["text"]
        )

        return result

    def repair(
        self,
        task,
        current_files,
        test_output,
        diagnosis,
    ):

        result = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
Repair this project.

ORIGINAL REQUIREMENTS:

{task}

CURRENT PROJECT:

{json.dumps(current_files, indent=2)}

PYTEST FAILURE:

{test_output}

TESTER DIAGNOSIS:

{json.dumps(diagnosis, indent=2)}

Return the COMPLETE corrected project.

Do not return only the changed file.
Return all project files.
""",
            max_new_tokens=2048,
            do_sample=False,
        )

        result["project"] = self._parse(
            result["text"]
        )

        return result

    def _parse(self, text):

        text = text.strip()

        if text.startswith("```json"):
            text = text[len("```json"):]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        data = json.loads(text)

        if "files" not in data:
            raise ValueError(
                "Developer response missing 'files'"
            )

        return data
