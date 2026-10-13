import argparse
import shutil
import subprocess
import time
from pathlib import Path

from llm.model_7b import LocalLLM7B
from agents.developer import DeveloperAgent
from agents.tester import TesterAgent

MAX_ATTEMPTS = 3


def clean_code(text):

    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def run_tests(workspace):

    start = time.time()

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "test_solution.py",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )

    elapsed = time.time() - start

    output = result.stdout + "\n" + result.stderr

    return result.returncode == 0, output, elapsed


def prepare_workspace(benchmark, workspace):

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True)

    shutil.copy(
        benchmark / "test_solution.py",
        workspace / "test_solution.py",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "benchmark",
        help="Benchmark name, for example bank_account",
    )

    args = parser.parse_args()

    benchmark = Path("benchmarks") / args.benchmark

    if not benchmark.exists():
        raise FileNotFoundError(
            f"Benchmark does not exist: {benchmark}"
        )

    task_file = benchmark / "task.txt"
    test_file = benchmark / "test_solution.py"

    if not task_file.exists():
        raise FileNotFoundError(task_file)

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    task = task_file.read_text()

    workspace = Path("workspaces") / args.benchmark

    prepare_workspace(
        benchmark,
        workspace,
    )

    print("=" * 60)
    print("PROJECT")
    print("=" * 60)
    print(args.benchmark)

    print()
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    llm = LocalLLM7B("cuda:0")

    developer = DeveloperAgent(llm)
    tester = TesterAgent(llm)

    current_code = None
    test_output = None

    total_generation_time = 0
    total_test_time = 0
    total_tokens = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print()
        print("=" * 60)
        print(f"DEVELOPER ATTEMPT {attempt}")
        print("=" * 60)

        if attempt == 1:

            result = developer.implement(task)

        else:

            print()
            print("-" * 60)
            print()
            print("-" * 60)
            print("TESTER AGENT DIAGNOSIS")
            print("-" * 60)
            
            diagnosis_result = tester.diagnose(
                task=task,
                current_code=current_code,
                test_output=test_output,
            )
            
            diagnosis = diagnosis_result["diagnosis"]
            
            total_generation_time += diagnosis_result["time"]
            total_tokens += diagnosis_result["tokens"]
            
            print("Problem:")
            print(diagnosis["problem"])
            
            print()
            print("Location:")
            print(diagnosis["location"])
            
            print()
            print("Cause:")
            print(diagnosis["cause"])
            
            print()
            print("Required change:")
            print(diagnosis["required_change"])
            
            print()
            print("-" * 60)
            print("DEVELOPER REPAIR")
            print("-" * 60)
            
            result = developer.fix(
                task=task,
                current_code=current_code,
                test_output=test_output,
                diagnosis=diagnosis,
            )

        total_generation_time += result["time"]
        total_tokens += result["tokens"]

        current_code = clean_code(result["text"])

        solution_file = workspace / "solution.py"

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

        passed, test_output, test_time = run_tests(
            workspace
        )

        total_test_time += test_time

        print(test_output)

        if passed:

            print()
            print("=" * 60)
            print("PROJECT PASSED")
            print("=" * 60)

            print(f"Attempts: {attempt}")
            print(
                f"LLM time: "
                f"{total_generation_time:.2f}s"
            )
            print(
                f"Test time: "
                f"{total_test_time:.2f}s"
            )
            print(
                f"Generated tokens: "
                f"{total_tokens}"
            )

            return

        print(
            "Tests failed. "
            "Sending feedback to Developer..."
        )

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
