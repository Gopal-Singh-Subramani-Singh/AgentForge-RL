import json


def extract_text(value):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if isinstance(
            value.get("text"),
            str,
        ):
            return value["text"]

    return str(value)


def clean_json(text):
    text = extract_text(
        text
    ).strip()

    fence = chr(96) * 3

    if fence in text:
        first = text.find(
            fence
        )

        start = text.find(
            "\n",
            first,
        )

        if start != -1:
            end = text.find(
                fence,
                start + 1,
            )

            if end != -1:
                text = text[
                    start + 1:end
                ].strip()

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
    ):
        text = text[
            start:end + 1
        ]

    return text


def parse_json(value):
    text = clean_json(
        value
    )

    try:
        return json.loads(
            text
        )

    except json.JSONDecodeError:
        return {}


class GenericRoleExecutor:
    def __init__(self, llm):
        self.llm = llm

    def project_context(
        self,
        files,
        max_chars=24000,
    ):
        chunks = []
        total = 0

        for name, content in (
            files.items()
        ):
            chunk = (
                f"\nFILE: {name}\n"
                f"{content}\n"
            )

            if (
                total + len(chunk)
                > max_chars
            ):
                break

            chunks.append(chunk)
            total += len(chunk)

        if not chunks:
            return (
                "(project is empty)"
            )

        return "\n".join(
            chunks
        )

    def developer(
        self,
        project_request,
        task,
        files,
        failure=None,
    ):
        context = (
            self.project_context(
                files
            )
        )

        failure_text = (
            failure
            if failure
            else "None"
        )

        prompt = f"""
PROJECT REQUEST:
{project_request}

CURRENT DEVELOPMENT TASK:
{task}

CURRENT PROJECT FILES:
{context}

LATEST TEST/BUILD FAILURE:
{failure_text}

You are the Developer agent.

Implement or repair the software necessary
for the current development task.

If a real test failure is supplied, inspect it
carefully and repair its root cause.

Rules:

- You may modify multiple files.
- Preserve already working functionality.
- Do not delete unrelated features.
- Use a sensible Python package structure.
- Prefer the Python standard library unless
  another dependency is genuinely required.
- Never modify tests merely to make them pass.
- Never return Markdown code fences.
- Return COMPLETE contents of every file changed.
- Do not return prose outside JSON.

Return exactly:

{{
  "summary": "what was implemented or repaired",
  "files": {{
    "relative/path.py": "complete file content"
  }}
}}
"""

        result = self.llm.generate(
            system_prompt=(
                "You are a senior Python "
                "software engineer."
            ),
            user_prompt=prompt,
            max_new_tokens=3200,
            do_sample=False,
        )

        return parse_json(
            result
        )

    def documentation(
        self,
        project_request,
        task,
        files,
    ):
        context = (
            self.project_context(
                files
            )
        )

        prompt = f"""
PROJECT REQUEST:
{project_request}

DOCUMENTATION TASK:
{task}

CURRENT PROJECT:
{context}

Write accurate documentation.

Rules:

- Document only functionality that exists.
- Include installation and usage when useful.
- Do not invent features.
- Return complete files.
- No Markdown fences surrounding JSON.

Return exactly:

{{
  "summary": "documentation update",
  "files": {{
    "README.md": "complete README contents"
  }}
}}
"""

        result = self.llm.generate(
            system_prompt=(
                "You are a technical "
                "documentation engineer."
            ),
            user_prompt=prompt,
            max_new_tokens=2400,
            do_sample=False,
        )

        return parse_json(
            result
        )

    def review(
        self,
        project_request,
        files,
        quality_output,
    ):
        context = (
            self.project_context(
                files
            )
        )

        prompt = f"""
PROJECT REQUEST:
{project_request}

PROJECT:
{context}

TEST / QUALITY RESULT:
{quality_output}

Perform the final review.

Approve only if the requested functionality
appears implemented and the quality checks pass.

Return exactly:

{{
  "approved": true,
  "score": 1.0,
  "issues": [],
  "reason": "..."
}}
"""

        result = self.llm.generate(
            system_prompt=(
                "You are a senior "
                "software reviewer."
            ),
            user_prompt=prompt,
            max_new_tokens=1200,
            do_sample=False,
        )

        return parse_json(
            result
        )
