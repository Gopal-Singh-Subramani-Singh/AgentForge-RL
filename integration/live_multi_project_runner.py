from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys

from llm.model_7b import LocalLLM7B
from integration.rl_scheduler import MARLPPOScheduler


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


PROJECTS = {
    "bank": {
        "name": "Bank Account",
        "requirements": """
Create app.py containing class BankAccount.

Requirements:
- __init__(owner, initial_balance=0)
- owner attribute
- balance starts at initial_balance
- negative initial balance raises ValueError
- deposit(amount):
    - amount must be > 0
    - otherwise raise ValueError
    - update and return balance
- withdraw(amount):
    - amount must be > 0
    - cannot exceed balance
    - invalid withdrawal raises ValueError
    - update and return balance
- get_balance() returns current balance

Do not create a CLI.
Do not use external libraries.
""",
        "tests": r'''
import pytest

from app import BankAccount


def test_initial_balance():
    a = BankAccount("Alice", 100)
    assert a.owner == "Alice"
    assert a.get_balance() == 100


def test_default_balance():
    a = BankAccount("Alice")
    assert a.get_balance() == 0


def test_negative_initial_balance():
    with pytest.raises(ValueError):
        BankAccount("Alice", -1)


def test_deposit():
    a = BankAccount("Alice", 10)
    assert a.deposit(25) == 35
    assert a.get_balance() == 35


@pytest.mark.parametrize("amount", [0, -1, -100])
def test_invalid_deposit(amount):
    a = BankAccount("Alice", 10)
    with pytest.raises(ValueError):
        a.deposit(amount)


def test_withdraw():
    a = BankAccount("Alice", 100)
    assert a.withdraw(30) == 70


def test_overdraft():
    a = BankAccount("Alice", 50)
    with pytest.raises(ValueError):
        a.withdraw(51)


@pytest.mark.parametrize("amount", [0, -1])
def test_invalid_withdraw(amount):
    a = BankAccount("Alice", 50)
    with pytest.raises(ValueError):
        a.withdraw(amount)


def test_multiple_operations():
    a = BankAccount("Alice", 100)
    a.deposit(50)
    a.withdraw(20)
    assert a.get_balance() == 130
''',
    },

    "todo": {
        "name": "Todo List",
        "requirements": """
Create app.py implementing an in-memory todo service.

Required module-level functions:

create_todo(title)
get_todo(todo_id)
list_todos()
complete_todo(todo_id)
delete_todo(todo_id)

Rules:
- Todos have integer id, title, and completed fields.
- IDs start at 1 and increase.
- title cannot be empty or whitespace.
- get_todo returns the matching todo.
- complete_todo marks completed=True.
- delete_todo removes and returns the todo.
- Unknown IDs raise KeyError.
- list_todos returns all todos.
- Do not use a database.
- Do not create a CLI.
""",
        "tests": r'''
import importlib
import pytest
import app


@pytest.fixture(autouse=True)
def fresh_module():
    importlib.reload(app)


def test_create():
    todo = app.create_todo("Write tests")
    assert todo["id"] == 1
    assert todo["title"] == "Write tests"
    assert todo["completed"] is False


def test_ids_increment():
    a = app.create_todo("A")
    b = app.create_todo("B")
    assert a["id"] == 1
    assert b["id"] == 2


def test_get():
    todo = app.create_todo("A")
    assert app.get_todo(todo["id"])["title"] == "A"


def test_list():
    app.create_todo("A")
    app.create_todo("B")
    assert len(app.list_todos()) == 2


def test_complete():
    todo = app.create_todo("A")
    result = app.complete_todo(todo["id"])
    assert result["completed"] is True


def test_delete():
    todo = app.create_todo("A")
    removed = app.delete_todo(todo["id"])
    assert removed["title"] == "A"
    assert app.list_todos() == []


def test_unknown():
    with pytest.raises(KeyError):
        app.get_todo(999)


def test_empty_title():
    with pytest.raises(ValueError):
        app.create_todo("   ")
''',
    },

    "inventory": {
        "name": "Inventory Manager",
        "requirements": """
Create app.py containing class Inventory.

Required behavior:
- Inventory starts empty.
- add_product(name, quantity=0)
- product names must be non-empty strings.
- initial product quantities must be non-negative integers.
- duplicate product names raise ValueError.
- stock_in(name, quantity) increases inventory.
- stock_in quantity must be a positive integer greater than 0.
- stock_out(name, quantity) decreases inventory.
- stock_out quantity must be a positive integer greater than 0.
- stock_out cannot make quantity negative.
- invalid quantities raise ValueError.
- unknown products raise KeyError.
- get_quantity(name) returns current quantity.
- list_products() returns a dictionary of product names to quantities.

Do not create a CLI.
Do not use external libraries.
""",
        "tests": r'''
import pytest
from app import Inventory


def test_empty():
    inv = Inventory()
    assert inv.list_products() == {}


def test_add():
    inv = Inventory()
    inv.add_product("Laptop", 3)
    assert inv.get_quantity("Laptop") == 3


def test_duplicate():
    inv = Inventory()
    inv.add_product("Laptop")
    with pytest.raises(ValueError):
        inv.add_product("Laptop")


def test_stock_in():
    inv = Inventory()
    inv.add_product("Mouse", 2)
    inv.stock_in("Mouse", 5)
    assert inv.get_quantity("Mouse") == 7


def test_stock_out():
    inv = Inventory()
    inv.add_product("Mouse", 10)
    inv.stock_out("Mouse", 4)
    assert inv.get_quantity("Mouse") == 6


def test_overdraw():
    inv = Inventory()
    inv.add_product("Mouse", 2)
    with pytest.raises(ValueError):
        inv.stock_out("Mouse", 3)


@pytest.mark.parametrize("value", [0, -1, 1.5, "2"])
def test_bad_stock_in(value):
    inv = Inventory()
    inv.add_product("Mouse", 2)
    with pytest.raises(ValueError):
        inv.stock_in("Mouse", value)


def test_unknown_product():
    inv = Inventory()
    with pytest.raises(KeyError):
        inv.get_quantity("Nope")
''',
    },

    "password": {
        "name": "Password Validator",
        "requirements": """
Create app.py implementing:

validate_password(password)

It returns a dictionary with:
- valid: bool
- errors: list[str]

A password is valid only when:
- it is a string
- at least 8 characters
- contains an uppercase letter
- contains a lowercase letter
- contains a digit
- contains at least one special character from !@#$%^&*

Return all violated rules in errors.
Do not raise errors for normal invalid passwords.
Do not use external libraries.
""",
        "tests": r'''
from app import validate_password


def test_valid():
    result = validate_password("Strong1!")
    assert result["valid"] is True
    assert result["errors"] == []


def test_short():
    result = validate_password("Aa1!")
    assert result["valid"] is False
    assert result["errors"]


def test_uppercase():
    assert validate_password("lowercase1!")["valid"] is False


def test_lowercase():
    assert validate_password("UPPERCASE1!")["valid"] is False


def test_digit():
    assert validate_password("Password!")["valid"] is False


def test_special():
    assert validate_password("Password1")["valid"] is False


def test_multiple_errors():
    result = validate_password("abc")
    assert result["valid"] is False
    assert len(result["errors"]) >= 3


def test_non_string():
    result = validate_password(12345)
    assert result["valid"] is False
    assert result["errors"]
''',
    },
}


def emit(event, **kwargs):
    payload = {
        "event": event,
        **kwargs,
    }

    print(
        json.dumps(payload),
        flush=True,
    )


def llm_text(result):
    """Normalize LocalLLM7B.generate() output to plain text."""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        # Support the formats commonly returned by our LLM wrapper.
        for key in (
            "text",
            "generated_text",
            "content",
            "response",
            "output",
        ):
            value = result.get(key)

            if isinstance(value, str):
                return value

        # Some wrappers return nested assistant messages.
        message = result.get("message")

        if isinstance(message, dict):
            content = message.get("content")

            if isinstance(content, str):
                return content

        raise TypeError(
            "Unsupported LLM dictionary output. "
            f"Available keys: {list(result.keys())}"
        )

    if isinstance(result, list):
        if not result:
            return ""

        return llm_text(result[0])

    raise TypeError(
        "Unsupported LLM output type: "
        f"{type(result).__name__}"
    )


def clean_code(text):
    text = llm_text(text).strip()

    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.S | re.I,
    )

    if match:
        return match.group(1).strip()

    return text


def run_tests(workspace):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        proc.returncode == 0,
        proc.stdout,
    )


def obs_vector(state):
    initial_tasks = 1.0

    return [
        state["dev_tasks"] / initial_tasks,
        state["untested"] / initial_tasks,
        state["bugs"] / initial_tasks,
        state["docs_progress"],
        float(state["docs_complete"]),
        float(state["review_rejected"]),
        2.0 / 4.0,
        1.0 - state["dev_tasks"] / initial_tasks,
        float(state["tests_passed"]),
        state["step"] / 20.0,
    ]


def generate_developer(llm, spec, previous_code, test_output):
    if not previous_code:
        prompt = f"""
You are the Developer agent in a software engineering team.

Implement the following requirements exactly.

REQUIREMENTS:
{spec["requirements"]}

Return ONLY the complete contents of app.py.
No markdown fences.
No explanation.
"""
    else:
        prompt = f"""
You are the Developer agent repairing app.py.

ORIGINAL REQUIREMENTS:
{spec["requirements"]}

CURRENT app.py:
{previous_code}

PYTEST FAILURE:
{test_output}

Return ONLY the complete corrected contents of app.py.

IMPORTANT REPAIR RULES:
- Fix the actual pytest failure, not just restate the previous code.
- Preserve the required public API.
- Re-read the ORIGINAL REQUIREMENTS before changing code.
- The failing test is authoritative.
- Read the exact failing input and assertion carefully.
- If pytest expects a boundary value such as 0 to raise ValueError,
  update the condition so that exact boundary is rejected.
- Do NOT modify tests.
- Do NOT repeat code that produces the same failing behavior.
- When requirements say to return ALL violated rules, accumulate every
  applicable error before returning.
- For password validation specifically, after confirming the input is a
  string, do NOT return early merely because the password is too short.
  Length, uppercase, lowercase, digit, and special-character rules must
  all be checked independently and all violated rules must be included
  in the errors list.

Do not invent extra architecture.
No markdown fences.
No explanation.
"""

    result = llm.generate(
        system_prompt=(
            "You are a precise senior Python developer. "
            "Follow the requested API exactly."
        ),
        user_prompt=prompt,
        max_new_tokens=1536,
        do_sample=False,
    )

    return clean_code(result)


def generate_docs(llm, spec):
    result = llm.generate(
        system_prompt=(
            "You write concise software documentation."
        ),
        user_prompt=f"""
Write a short README.md for this Python project.

PROJECT:
{spec["name"]}

REQUIREMENTS:
{spec["requirements"]}

Include:
- purpose
- API
- simple usage example
- how to run pytest

Return Markdown only.
""",
        max_new_tokens=1024,
        do_sample=False,
    )

    return llm_text(result).strip()


def tester_diagnosis(llm, spec, code, test_output):
    result = llm.generate(
        system_prompt=(
            "You are a software test engineer."
        ),
        user_prompt=f"""
Analyze this failing Python implementation.

REQUIREMENTS:
{spec["requirements"]}

CURRENT CODE:
{code}

PYTEST OUTPUT:
{test_output}

Explain the root cause and exact required correction.

Do NOT provide replacement source code.
Do NOT suggest changing the tests.
Base the diagnosis strictly on the original requirements and pytest failure.
State the behavioral change the Developer must make.

When a pytest failure involves a boundary value such as 0:
- state explicitly whether 0 must be accepted or rejected;
- identify the exact comparison that must change, such as < 0 versus <= 0.

Never recommend replacing a condition with an identical condition.

For validation functions that must return all violated rules, explicitly
identify any premature return that prevents later rules from being checked.

Be concise.
""",
        max_new_tokens=512,
        do_sample=False,
    )

    return llm_text(result).strip()


def reviewer_decision(llm, spec, code):
    result = llm.generate(
        system_prompt=(
            "You are a strict senior code reviewer."
        ),
        user_prompt=f"""
Review this implementation against the requirements.

REQUIREMENTS:
{spec["requirements"]}

CODE:
{code}

Automated tests have passed.

Return exactly one line:
APPROVED
or
REJECTED: <brief reason>
""",
        max_new_tokens=256,
        do_sample=False,
    )

    text = llm_text(result).strip()

    approved = (
        text.upper().startswith("APPROVED")
    )

    return approved, text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        choices=sorted(PROJECTS),
        required=True,
    )

    args = parser.parse_args()

    spec = PROJECTS[args.project]

    workspace = (
        Path("workspaces")
        / f"live_{args.project}"
    )

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    (workspace / "test_app.py").write_text(
        spec["tests"]
    )

    emit(
        "starting",
        project=args.project,
        name=spec["name"],
    )

    emit(
        "status",
        message="Loading Qwen2.5-Coder-7B...",
    )

    llm = LocalLLM7B()

    emit(
        "status",
        message="Loading MARL PPO scheduler...",
    )

    scheduler = MARLPPOScheduler(
        checkpoint=(
            "results/checkpoints/"
            "marl_ppo_alpha_0_25"
        ),
        max_resources=2,
    )

    state = {
        "step": 0,
        "dev_tasks": 1,
        "untested": 0,
        "bugs": 0,
        "docs_progress": 0.0,
        "docs_complete": False,
        "review_rejected": False,
        "tests_passed": False,
        "complete": False,
    }

    current_code = ""
    last_test_output = ""

    try:
        while (
            state["step"] < 20
            and not state["complete"]
        ):
            obs = obs_vector(state)

            decision = scheduler.decide(obs)

            requested = decision[
                "requested"
            ]

            allocated = decision[
                "allocated"
            ]

            emit(
                "allocation",
                step=state["step"],
                observation=obs,
                requested=requested,
                allocated=allocated,
                state=state.copy(),
            )

            state[
                "review_rejected"
            ] = False

            for agent in allocated:
                emit(
                    "agent_start",
                    agent=agent,
                    step=state["step"],
                )

                if agent == "developer":
                    if (
                        state["dev_tasks"] > 0
                        or state["bugs"] > 0
                    ):
                        current_code = (
                            generate_developer(
                                llm,
                                spec,
                                current_code,
                                last_test_output,
                            )
                        )

                        (
                            workspace / "app.py"
                        ).write_text(
                            current_code
                        )

                        state[
                            "dev_tasks"
                        ] = 0

                        state[
                            "bugs"
                        ] = 0

                        state[
                            "untested"
                        ] = 1

                        state[
                            "tests_passed"
                        ] = False

                        emit(
                            "artifact",
                            agent="developer",
                            file="app.py",
                            content=current_code,
                        )

                elif agent == "documentation":
                    if not state[
                        "docs_complete"
                    ]:
                        docs = generate_docs(
                            llm,
                            spec,
                        )

                        (
                            workspace / "README.md"
                        ).write_text(
                            docs
                        )

                        state[
                            "docs_progress"
                        ] = 1.0

                        state[
                            "docs_complete"
                        ] = True

                        emit(
                            "artifact",
                            agent="documentation",
                            file="README.md",
                            content=docs,
                        )

                elif agent == "tester":
                    if (
                        state["untested"] > 0
                        and current_code
                    ):
                        (
                            passed,
                            output,
                        ) = run_tests(
                            workspace
                        )

                        last_test_output = (
                            output
                        )

                        emit(
                            "tests",
                            passed=passed,
                            output=output,
                        )

                        if passed:
                            state[
                                "untested"
                            ] = 0

                            state[
                                "bugs"
                            ] = 0

                            state[
                                "tests_passed"
                            ] = True

                        else:
                            diagnosis = (
                                tester_diagnosis(
                                    llm,
                                    spec,
                                    current_code,
                                    output,
                                )
                            )

                            state[
                                "untested"
                            ] = 0

                            state[
                                "bugs"
                            ] = 1

                            emit(
                                "diagnosis",
                                text=diagnosis,
                            )

                            last_test_output += (
                                "\n\nTESTER DIAGNOSIS:\n"
                                + diagnosis
                            )

                elif agent == "reviewer":
                    ready = (
                        state[
                            "dev_tasks"
                        ] == 0
                        and
                        state[
                            "untested"
                        ] == 0
                        and
                        state[
                            "bugs"
                        ] == 0
                        and
                        state[
                            "docs_complete"
                        ]
                        and
                        state[
                            "tests_passed"
                        ]
                    )

                    if ready:
                        (
                            approved,
                            review,
                        ) = reviewer_decision(
                            llm,
                            spec,
                            current_code,
                        )

                        emit(
                            "review",
                            approved=approved,
                            text=review,
                        )

                        if approved:
                            state[
                                "complete"
                            ] = True

                    else:
                        state[
                            "review_rejected"
                        ] = True

                        emit(
                            "review",
                            approved=False,
                            text=(
                                "Project is not ready "
                                "for final review."
                            ),
                        )

                emit(
                    "agent_end",
                    agent=agent,
                    step=state["step"],
                )

            state["step"] += 1

        if state["complete"]:
            emit(
                "completed",
                success=True,
                steps=state["step"],
                workspace=str(workspace),
            )
        else:
            emit(
                "completed",
                success=False,
                steps=state["step"],
                workspace=str(workspace),
            )

    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
