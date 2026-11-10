import ast
import os
import subprocess
from pathlib import Path


class PythonProjectAdapter:
    def __init__(self, workspace):
        self.workspace = Path(
            workspace
        ).resolve()

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

    def path(self, relative_path):
        relative_path = (
            relative_path
            .replace("\\", "/")
            .lstrip("/")
        )

        if ".." in Path(
            relative_path
        ).parts:
            raise ValueError(
                "Path may not escape workspace."
            )

        path = (
            self.workspace
            / relative_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def write_file(
        self,
        relative_path,
        content,
    ):
        self.path(
            relative_path
        ).write_text(
            content,
            encoding="utf-8",
        )

    def read_file(
        self,
        relative_path,
    ):
        path = self.path(
            relative_path
        )

        if not path.exists():
            return ""

        return path.read_text(
            encoding="utf-8",
        )

    def files(self):
        result = {}

        ignored = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
        }

        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue

            if any(
                part in ignored
                for part in path.parts
            ):
                continue

            try:
                relative = str(
                    path.relative_to(
                        self.workspace
                    )
                )

                result[relative] = (
                    path.read_text(
                        encoding="utf-8",
                    )
                )

            except Exception:
                pass

        return result

    def python_files(self):
        return list(
            self.workspace.rglob(
                "*.py"
            )
        )

    def syntax_check(self):
        errors = []

        for path in self.python_files():
            try:
                source = path.read_text(
                    encoding="utf-8",
                )

                ast.parse(
                    source,
                    filename=str(path),
                )

            except SyntaxError as exc:
                errors.append(
                    f"{path}: {exc}"
                )

        if errors:
            return (
                False,
                "\n".join(errors),
            )

        return (
            True,
            "Python syntax check passed.",
        )

    def pytest(
        self,
        target=None,
    ):
        command = [
            "python",
            "-m",
            "pytest",
            "-q",
        ]

        if target:
            command.append(
                str(target)
            )

        env = os.environ.copy()

        # Tests may chdir into tmp_path for clean
        # persistent storage. Keep the generated
        # project importable from subprocesses.
        existing = env.get(
            "PYTHONPATH",
            "",
        )

        env["PYTHONPATH"] = (
            str(self.workspace)
            + (
                os.pathsep + existing
                if existing
                else ""
            )
        )

        try:
            result = subprocess.run(
                command,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

        except subprocess.TimeoutExpired:
            return (
                False,
                "pytest timed out after 120 seconds.",
            )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        ).strip()

        return (
            result.returncode == 0,
            output,
        )

    def quality_gate(
        self,
        test_file=None,
    ):
        syntax_ok, syntax_output = (
            self.syntax_check()
        )

        if not syntax_ok:
            return {
                "passed": False,
                "stage": "syntax",
                "output": syntax_output,
            }

        test_files = list(
            self.workspace.rglob(
                "test_*.py"
            )
        )

        if not test_files:
            return {
                "passed": False,
                "stage": "pytest",
                "output": (
                    "No pytest tests exist."
                ),
            }

        passed, output = self.pytest(
            target=test_file
        )

        return {
            "passed": passed,
            "stage": (
                "pytest-task"
                if test_file
                else "pytest-full"
            ),
            "output": output,
        }
