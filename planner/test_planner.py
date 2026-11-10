import json

from llm.model_7b import LocalLLM7B
from planner import PlannerAgent


PROJECT = """
Build a Python todo management module.

Requirements:
- Create todos with unique IDs.
- Retrieve a todo by ID.
- List all todos.
- Mark todos complete.
- Delete todos.
- Handle missing IDs correctly.
- Include automated tests.
- Include documentation.
"""


def main():
    print("Loading 7B planner model...")
    llm = LocalLLM7B()

    planner = PlannerAgent(llm)

    print("\nGenerating plan...\n")

    plan = planner.plan(PROJECT)

    print(
        json.dumps(
            plan,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
