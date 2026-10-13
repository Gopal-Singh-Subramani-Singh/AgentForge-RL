import shutil
import subprocess
from pathlib import Path

from llm.model_7b import LocalLLM7B
from agents.developer import DeveloperAgent


TASK = """
Create a function:

    fibonacci(n)

Requirements:

- fibonacci(0) must return 0
- fibonacci(1) must return 1
- fibonacci(2) must return 1
- fibonacci(n) must return the nth Fibonacci number
- negative input must raise ValueError
"""


WORKSPACE = Path("workspaces/fibonacci")
TEST_SOURCE = Path("benchmarks/fibonacci/test_solution.py")

MAX_ATTEMPTS = 3


def clean_code(text):
    """
    Remove Markdown fences if the model ignores our instruction.
    """

    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def prepare_workspace():

    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)

    WORKSPACE.mkdir(parents=True)

    shutil.copy(
        TEST_SOURCE,
        WORKSPACE / "test_solution.py",
    )


def run_tests():

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "test_solution.py",
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout + "\n" + result.stderr

    return result.returncode == 0, output


def main():

    prepare_workspace()

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    llm = LocalLLM7B("cuda:0")
    developer = DeveloperAgent(llm)

    current_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print()
        print("=" * 60)
        print(f"DEVELOPER ATTEMPT {attempt}")
        print("=" * 60)

        if attempt == 1:

            result = developer.implement(TASK)

        else:

            print()
            print("-" * 60)
            print("DEBUG ANALYSIS")
            print("-" * 60)
            
            diagnosis_result = developer.analyze_failure(
                task=TASK,
                current_code=current_code,
                test_output=test_output,
            )
            
            diagnosis = diagnosis_result["text"]
            
            print(diagnosis)
            
            print()
            print("-" * 60)
            print("APPLYING REPAIR")
            print("-" * 60)
            
            result = developer.fix(
                task=TASK,
                current_code=current_code,
                test_output=test_output,
                diagnosis=diagnosis,
            )

        current_code = clean_code(result["text"])

        solution_file = WORKSPACE / "solution.py"

        solution_file.write_text(current_code)

        print(current_code)

        print()
        print(
            f"Generation time: "
            f"{result['time']:.2f}s"
        )

        print()
        print("=" * 60)
        print("TESTER")
        print("=" * 60)

        passed, test_output = run_tests()

        print(test_output)

        if passed:

            print()
            print("=" * 60)
            print("PROJECT PASSED")
            print("=" * 60)

            print(f"Attempts: {attempt}")

            return

        print("Tests failed. Sending feedback to Developer...")

    print()
    print("=" * 60)
    print("PROJECT FAILED")
    print("=" * 60)

    print(
        f"Developer could not pass all tests "
        f"after {MAX_ATTEMPTS} attempts."
    )


if __name__ == "__main__":
    main()
