import json
import subprocess
from pathlib import Path

import numpy as np

from agents.developer import DeveloperAgent
from agents.tester import TesterAgent
from agents.documentation import DocumentationAgent
from agents.reviewer import ReviewerAgent

from integration.rl_scheduler import MARLPPOScheduler
from llm.model_7b import LocalLLM7B


TASK = """
Build a Python Todo service in app.py.

Requirements:

1. Maintain an in-memory collection of todo items.
2. Each todo has:
   - id: integer
   - title: string
   - completed: boolean
3. Implement these functions:

   create_todo(title)
   get_todo(todo_id)
   list_todos()
   complete_todo(todo_id)
   delete_todo(todo_id)

4. create_todo:
   - creates a new todo
   - assigns a unique integer id
   - completed defaults to False
   - returns the created todo dictionary

5. get_todo:
   - returns the todo with the requested id
   - returns None if it does not exist

6. list_todos:
   - returns all todos as a list

7. complete_todo:
   - marks the requested todo completed
   - returns the updated todo
   - returns None if the id does not exist

8. delete_todo:
   - removes the requested todo
   - returns True if deleted
   - returns False if the id does not exist

The module must be self-contained and must not require a database.
"""


TEST_FILE = '''
import importlib
import app


def setup_function():
    importlib.reload(app)


def test_create_todo():
    todo = app.create_todo("Write code")

    assert todo["id"] == 1
    assert todo["title"] == "Write code"
    assert todo["completed"] is False


def test_unique_ids():
    a = app.create_todo("A")
    b = app.create_todo("B")

    assert a["id"] != b["id"]


def test_get_todo():
    todo = app.create_todo("Read")

    result = app.get_todo(todo["id"])

    assert result == todo


def test_missing_todo():
    assert app.get_todo(999) is None


def test_list_todos():
    app.create_todo("A")
    app.create_todo("B")

    result = app.list_todos()

    assert len(result) == 2


def test_complete_todo():
    todo = app.create_todo("Finish")

    result = app.complete_todo(
        todo["id"]
    )

    assert result["completed"] is True


def test_complete_missing():
    assert app.complete_todo(999) is None


def test_delete_todo():
    todo = app.create_todo("Delete")

    assert app.delete_todo(
        todo["id"]
    ) is True

    assert app.get_todo(
        todo["id"]
    ) is None


def test_delete_missing():
    assert app.delete_todo(999) is False
'''


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


class RealProjectState:

    def __init__(
        self,
        initial_tasks=5,
        num_resources=2,
        max_steps=20,
    ):

        self.initial_tasks = initial_tasks
        self.num_resources = num_resources
        self.max_steps = max_steps

        self.step = 0

        self.dev_tasks = initial_tasks
        self.untested_work = 0
        self.bugs = 0

        self.docs_progress = 0.0
        self.docs_complete = False

        self.review_rejected = False
        self.review_complete = False

        self.completed_dev_tasks = 0
        self.completed_tests = 0

    def observation(self):

        dev_progress = min(
            self.completed_dev_tasks
            / self.initial_tasks,
            1.0,
        )

        test_progress = min(
            self.completed_tests
            / self.initial_tasks,
            1.0,
        )

        normalized_time = min(
            self.step
            / self.max_steps,
            1.0,
        )

        return np.array(
            [
                self.dev_tasks
                / self.initial_tasks,

                self.untested_work
                / self.initial_tasks,

                self.bugs
                / self.initial_tasks,

                self.docs_progress,

                float(
                    self.docs_complete
                ),

                float(
                    self.review_rejected
                ),

                self.num_resources
                / 4.0,

                dev_progress,

                test_progress,

                normalized_time,
            ],
            dtype=np.float32,
        )

    def summary(self):

        return {
            "step":
                self.step,

            "dev_tasks":
                self.dev_tasks,

            "untested_work":
                self.untested_work,

            "bugs":
                self.bugs,

            "docs_progress":
                round(
                    self.docs_progress,
                    2,
                ),

            "docs_complete":
                self.docs_complete,

            "review_rejected":
                self.review_rejected,

            "review_complete":
                self.review_complete,
        }


class RLTeamRunner:

    def __init__(
        self,
        workspace,
        resources=2,
        max_steps=20,
    ):

        self.workspace = Path(
            workspace
        ).resolve()

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.task = TASK

        self.state = RealProjectState(
            num_resources=resources,
            max_steps=max_steps,
        )

        self.max_steps = max_steps

        #
        # Always install the benchmark tests.
        #
        self.test_path = (
            self.workspace
            / "test_app.py"
        )

        self.test_path.write_text(
            TEST_FILE
        )

        self.app_path = (
            self.workspace
            / "app.py"
        )

        self.readme_path = (
            self.workspace
            / "README.md"
        )

        self.last_test_output = (
            "Tests have not run yet."
        )

        self.last_diagnosis = None

        print(
            "Loading MARL PPO scheduler..."
        )

        self.scheduler = (
            MARLPPOScheduler(
                max_resources=resources
            )
        )

        print(
            "Loading shared 7B LLM..."
        )

        self.llm = LocalLLM7B()

        self.agents = {
            "developer":
                DeveloperAgent(
                    self.llm
                ),

            "tester":
                TesterAgent(
                    self.llm
                ),

            "documentation":
                DocumentationAgent(
                    self.llm
                ),

            "reviewer":
                ReviewerAgent(
                    self.llm
                ),
        }

    def extract_text(
        self,
        result,
    ):

        if isinstance(
            result,
            str,
        ):
            return result.strip()

        if isinstance(
            result,
            dict,
        ):
            return result.get(
                "text",
                "",
            ).strip()

        return str(
            result
        ).strip()

    def clean_python_source(
        self,
        text,
    ):
        text = text.strip()

        if text.startswith("```python"):
            text = text[len("```python"):]

        elif text.startswith("```py"):
            text = text[len("```py"):]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def project_files(self):

        files = {}

        for path in (
            self.workspace
            .glob("*")
        ):

            if (
                path.is_file()
                and
                path.suffix
                in {
                    ".py",
                    ".md",
                    ".txt",
                }
            ):
                try:
                    files[
                        path.name
                    ] = path.read_text()

                except Exception:
                    pass

        return files

    def run_pytest(self):

        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
            ],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        ).strip()

        self.last_test_output = (
            output
        )

        passed = (
            result.returncode == 0
        )

        return passed, output

    def run_developer(self):

        print()
        print(
            "[Developer] executing"
        )

        developer = self.agents[
            "developer"
        ]

        #
        # Repair path.
        #
        if (
            self.app_path.exists()
            and
            self.last_diagnosis
            is not None
        ):

            print(
                "Mode: repair"
            )

            current_code = (
                self.app_path
                .read_text()
            )

            result = developer.fix(
                task=self.task,
                current_code=current_code,
                test_output=(
                    self.last_test_output
                ),
                diagnosis=(
                    self.last_diagnosis
                ),
            )

            code = self.clean_python_source(
                self.extract_text(
                    result
                )
            )

            self.app_path.write_text(
                code
            )

            self.last_diagnosis = None

            if self.state.bugs > 0:
                self.state.bugs -= 1

            self.state.untested_work = max(
                self.state.untested_work,
                1,
            )

            print(
                "Updated app.py"
            )

            return

        #
        # Initial implementation.
        #
        if not self.app_path.exists():

            print(
                "Mode: implementation"
            )

            result = (
                developer.implement(
                    self.task
                )
            )

            code = self.clean_python_source(
                self.extract_text(
                    result
                )
            )

            self.app_path.write_text(
                code
            )

            self.state.completed_dev_tasks = (
                self.state.initial_tasks
            )

            self.state.dev_tasks = 0
            self.state.untested_work = 1

            print(
                "Created app.py"
            )

            return

        #
        # Nothing new to implement.
        #
        print(
            "No pending developer work."
        )

    def run_tester(self):

        print()
        print(
            "[Tester] executing"
        )

        if not self.app_path.exists():

            print(
                "No implementation exists yet."
            )

            return

        passed, output = (
            self.run_pytest()
        )

        print()
        print(
            "pytest:"
        )

        print(output)

        if passed:

            print(
                "Tests passed."
            )

            self.state.untested_work = 0
            self.state.bugs = 0

            self.state.completed_tests = (
                self.state.initial_tasks
            )

            self.last_diagnosis = None

            return

        print(
            "Tests failed; requesting "
            "Tester diagnosis."
        )

        current_code = (
            self.app_path
            .read_text()
        )

        result = (
            self.agents[
                "tester"
            ].diagnose(
                task=self.task,
                current_code=current_code,
                test_output=output,
            )
        )

        diagnosis = result.get(
            "diagnosis",
            {}
        )

        self.last_diagnosis = (
            diagnosis
        )

        self.state.bugs = max(
            self.state.bugs,
            1,
        )

        self.state.untested_work = 0

        print()
        print(
            "Diagnosis:"
        )

        print(
            json.dumps(
                diagnosis,
                indent=2,
            )
        )

    def run_documentation(self):

        print()
        print(
            "[Documentation] executing"
        )

        if not self.app_path.exists():

            print(
                "Implementation not ready; "
                "documentation deferred."
            )

            return

        files = (
            self.project_files()
        )

        result = (
            self.agents[
                "documentation"
            ].create(
                task=self.task,
                project_files=files,
            )
        )

        readme = self.extract_text(
            result
        )

        self.readme_path.write_text(
            readme
        )

        self.state.docs_progress = 1.0
        self.state.docs_complete = True

        print(
            "Created README.md"
        )

    def run_reviewer(self):

        print()
        print(
            "[Reviewer] executing"
        )

        #
        # Run real tests immediately before review.
        #
        if self.app_path.exists():

            passed, output = (
                self.run_pytest()
            )

        else:

            passed = False
            output = (
                "app.py does not exist."
            )

        result = (
            self.agents[
                "reviewer"
            ].review(
                task=self.task,
                project_files=(
                    self.project_files()
                ),
                test_output=output,
            )
        )

        review = result.get(
            "review",
            {}
        )

        print()
        print(
            "Review:"
        )

        print(
            json.dumps(
                review,
                indent=2,
            )
        )

        approved = bool(
            review.get(
                "approved",
                False,
            )
        )

        #
        # We require both reviewer approval
        # and actual pytest success.
        #
        if approved and passed:

            self.state.review_complete = True
            self.state.review_rejected = False

            return

        self.state.review_complete = False
        self.state.review_rejected = True

        send_back = review.get(
            "send_back_to",
            "none",
        )

        if not passed:

            self.state.bugs = max(
                self.state.bugs,
                1,
            )

            self.state.untested_work = 1

        if (
            send_back
            == "documentation"
        ):
            self.state.docs_complete = False
            self.state.docs_progress = 0.75

    def execute_role(
        self,
        role,
    ):

        if role == "developer":
            self.run_developer()

        elif role == "tester":
            self.run_tester()

        elif role == "documentation":
            self.run_documentation()

        elif role == "reviewer":
            self.run_reviewer()

    def run(self):

        print()
        print("=" * 72)
        print(
            "RL-SCHEDULED REAL 7B LLM TEAM"
        )
        print("=" * 72)

        while (
            self.state.step
            < self.max_steps
        ):

            print()
            print("=" * 72)

            print(
                f"STEP {self.state.step}"
            )

            print(
                "State:",
                self.state.summary(),
            )

            observation = (
                self.state.observation()
            )

            decision = (
                self.scheduler.decide(
                    observation
                )
            )

            print(
                "Requested:",
                decision[
                    "requested"
                ],
            )

            print(
                "Allocated:",
                decision[
                    "allocated"
                ],
            )

            for role in (
                decision[
                    "allocated"
                ]
            ):

                self.execute_role(
                    role
                )

                if (
                    self.state
                    .review_complete
                ):
                    break

            if (
                self.state
                .review_complete
            ):

                print()
                print("=" * 72)
                print(
                    "PROJECT COMPLETED"
                )
                print("=" * 72)

                break

            self.state.step += 1

        if not (
            self.state.review_complete
        ):

            print()
            print("=" * 72)
            print(
                "MAX STEPS REACHED"
            )
            print("=" * 72)

        print()
        print(
            "Final state:",
            self.state.summary(),
        )

        print()
        print(
            "Workspace:",
            self.workspace,
        )

        self.scheduler.close()


def main():

    runner = RLTeamRunner(
        workspace=(
            "workspaces/"
            "rl_todo_demo"
        ),
        resources=2,
        max_steps=20,
    )

    runner.run()


if __name__ == "__main__":
    main()
