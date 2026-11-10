import re


def _text(value):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if isinstance(
            value.get("text"),
            str,
        ):
            return value["text"]

    return str(value)


def _clean_python(value):
    text = _text(
        value
    ).strip()

    matches = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    if matches:
        text = matches[
            0
        ].strip()

    lines = text.splitlines()

    while lines and (
        lines[0]
        .strip()
        .lower()
        .startswith("here")
        or
        lines[0]
        .strip()
        .lower()
        .startswith(
            "test file"
        )
    ):
        lines.pop(0)

    return "\n".join(
        lines
    ).strip()


def generate_test(
    llm,
    project_request,
    task_id,
    description,
    files,
    other_test_tasks=None,
    attempt=1,
    previous_failure=None,
):
    context_parts = []
    total = 0

    for name, content in (
        files.items()
    ):
        block = (
            f"\nFILE: {name}\n"
            f"{content}\n"
        )

        if (
            total + len(block)
            > 20000
        ):
            break

        context_parts.append(
            block
        )

        total += len(
            block
        )

    context = (
        "\n".join(
            context_parts
        )
        if context_parts
        else "(project is empty)"
    )

    exclusions = ""

    if other_test_tasks:
        exclusions = "\n".join(
            f"- {item}"
            for item
            in other_test_tasks
        )

    retry_text = (
        previous_failure
        if previous_failure
        else "None"
    )

    prompt = f"""
PROJECT REQUEST:
{project_request}

CURRENT TEST TASK:
{description}

TEST GENERATION ATTEMPT:
{attempt}

OTHER TESTER TASKS:
{exclusions or "(none)"}

PREVIOUS FAILURE IF THIS IS A RETRY:
{retry_text}

CURRENT PROJECT:
{context}

You are responsible ONLY for the CURRENT TEST TASK.

Every Tester task has an isolated responsibility.

DO NOT test functionality assigned to another
Tester task.

If this is a retry, do not simply reproduce a
previous problematic test. Use a simpler public
behavior test that remains within the current
task specification.

Rules:

- Output Python source only.
- No JSON.
- No prose.
- No Markdown fences.
- Use pytest.
- Test ONLY the current task.
- Maximum 3 test functions.
- Do NOT duplicate another Tester task.
- Do NOT invent requirements.
- Do NOT require __str__ or __repr__ formatting
  unless explicitly requested.
- Do NOT require arbitrary exact display formatting
  unless explicitly required by the project request.
- Do NOT inspect private implementation state.
- Prefer public functional behavior.
- Tests must be deterministic.
- Import everything required.
- Use temporary files/directories when persistence
  needs to be tested.
- Do not depend on execution order between tests.

Return only valid Python source.
"""

    result = llm.generate(
        system_prompt=(
            "You are a precise Python "
            "test engineer. Strictly isolate "
            "each test task from all other "
            "test tasks."
        ),
        user_prompt=prompt,
        max_new_tokens=1900,
        do_sample=False,
    )

    source = _clean_python(
        result
    )

    # Every generated test runs from its own
    # temporary working directory. This prevents
    # CLI/database/JSON state from leaking from
    # one pytest test into another.
    isolation = """
import pytest

@pytest.fixture(autouse=True)
def _devteam_isolated_workspace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

"""

    if (
        "_devteam_isolated_workspace"
        not in source
    ):
        source = (
            isolation
            + "\n"
            + source
        )

    filename = (
        "tests/"
        + f"test_{task_id.lower()}.py"
    )

    return filename, source
