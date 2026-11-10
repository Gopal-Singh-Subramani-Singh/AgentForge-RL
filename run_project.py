import argparse
import shutil
import subprocess
from pathlib import Path
from agents.reviewer import ReviewerAgent
from llm.model_7b import LocalLLM7B
from agents.project_developer import ProjectDeveloperAgent
from agents.tester import TesterAgent
from agents.documentation import DocumentationAgent

MAX_ATTEMPTS = 3


def write_project(workspace, files):

    for filename, contents in files.items():

        path = workspace / filename

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(contents)


def run_tests(workspace):

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "test_app.py",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = (
        result.stdout +
        "\n" +
        result.stderr
    )

    return result.returncode == 0, output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "benchmark",
        help="Benchmark name",
    )

    args = parser.parse_args()

    benchmark = (
        Path("benchmarks") /
        args.benchmark
    )

    task = (
        benchmark /
        "task.txt"
    ).read_text()

    workspace = (
        Path("workspaces") /
        args.benchmark
    )

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(
        parents=True
    )

    shutil.copy(
        benchmark / "test_app.py",
        workspace / "test_app.py",
    )

    print("=" * 60)
    print("PROJECT:", args.benchmark)
    print("=" * 60)

    llm = LocalLLM7B("cuda:0")

    developer = ProjectDeveloperAgent(llm)
    tester = TesterAgent(llm)
    documentation = DocumentationAgent(llm)
    reviewer = ReviewerAgent(llm)

    current_files = None
    test_output = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        print()
        print("=" * 60)
        print(
            f"DEVELOPER ATTEMPT {attempt}"
        )
        print("=" * 60)

        if attempt == 1:

            result = developer.implement(
                task
            )

        else:

            diagnosis_result = (
                tester.diagnose(
                    task=task,
                    current_code=str(
                        current_files
                    ),
                    test_output=test_output,
                )
            )

            diagnosis = (
                diagnosis_result[
                    "diagnosis"
                ]
            )

            print()
            print(
                "TESTER DIAGNOSIS:"
            )

            print(diagnosis)

            result = developer.repair(
                task=task,
                current_files=current_files,
                test_output=test_output,
                diagnosis=diagnosis,
            )

        current_files = (
            result["project"]["files"]
        )

        write_project(
            workspace,
            current_files,
        )

        print()
        print("Generated files:")

        for filename in current_files:
            print(" -", filename)

        print()
        print("=" * 60)
        print("PYTEST")
        print("=" * 60)

        passed, test_output = (
            run_tests(workspace)
        )

        print(test_output)

        if passed:
        
            print()
            print("=" * 60)
            print("CODE PASSED")
            print("=" * 60)
        
            print(f"Developer attempts: {attempt}")
        
            print()
            print("=" * 60)
            print("DOCUMENTATION AGENT")
            print("=" * 60)
        
            docs_result = documentation.create(
                task=task,
                project_files=current_files,
            )
        
            readme = docs_result["text"].strip()
        
            if readme.startswith("```markdown"):
                readme = readme[len("```markdown"):]
        
            elif readme.startswith("```"):
                readme = readme[3:]
        
            if readme.endswith("```"):
                readme = readme[:-3]
        
            readme = readme.strip()
        
            (workspace / "README.md").write_text(
                readme
            )
        
            current_files["README.md"] = readme
        
            print("Created README.md")
        
            print()
            print("=" * 60)
            print()
            print("=" * 60)
            print("REVIEWER AGENT")
            print("=" * 60)
            
            review_result = reviewer.review(
                task=task,
                project_files=current_files,
                test_output=test_output,
            )
            
            review = review_result["review"]
            
            print("Approved:", review["approved"])
            print("Score:", review["score"])
            
            print()
            print("Issues:")
            
            if review["issues"]:
                for issue in review["issues"]:
                    print(" -", issue)
            else:
                print(" - None")
            
            print()
            print(
                "Send back to:",
                review["send_back_to"]
            )
            
            print()
            print(
                "Reason:",
                review["reason"]
            )
            
            if review["approved"]:
            
                print()
                print("=" * 60)
                print("PROJECT APPROVED")
                print("=" * 60)
            
                print("Generated files:")
            
                for filename in current_files:
                    print(" -", filename)
            
                return
            
            print()
            print("=" * 60)
            print("PROJECT REJECTED")
            print("=" * 60)
            
            print(
                "Reviewer requested changes from:",
                review["send_back_to"]
            )
            
            return

        print(
            "Tests failed. "
            "Sending to Tester Agent..."
        )

    print()
    print("=" * 60)
    print("PROJECT FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
