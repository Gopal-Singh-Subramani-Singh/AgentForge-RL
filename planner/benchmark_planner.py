import json
from collections import Counter

from llm.model_7b import LocalLLM7B
from planner import PlannerAgent


PROJECTS = {
    "level1_bank": """
Build a Python bank account module.

Requirements:
- BankAccount class.
- Owner and balance.
- Deposit money.
- Withdraw money.
- Prevent overdrafts.
- Reject invalid amounts.
- Automated tests.
- Documentation.
""",

    "level2_todo": """
Build a Python todo management module.

Requirements:
- Create todos with unique IDs.
- Retrieve todos by ID.
- List all todos.
- Mark todos complete.
- Delete todos.
- Handle missing IDs.
- Automated tests.
- Documentation.
""",

    "level3_fastapi": """
Build a FastAPI CRUD service for notes.

Requirements:
- FastAPI application.
- Note model with id, title, and content.
- Create note endpoint.
- Get note endpoint.
- List notes endpoint.
- Update note endpoint.
- Delete note endpoint.
- Input validation.
- Correct 404 handling.
- Automated API tests.
- README documentation.
""",

    "level4_auth_api": """
Build an authentication REST API.

Requirements:
- User registration.
- Unique usernames.
- Password hashing.
- User login.
- Token generation.
- Authentication-protected endpoint.
- Invalid credential handling.
- Persistent user storage.
- Automated tests for registration, login, and authentication.
- Security-related edge case tests.
- API documentation.
""",

    "level5_issue_tracker": """
Build a small issue-tracking backend.

Requirements:
- Users.
- Projects.
- Issues.
- Comments on issues.
- Create and retrieve projects.
- Create, update, and close issues.
- Assign issues to users.
- Add comments to issues.
- Persistent storage.
- Input validation.
- Missing-resource handling.
- Automated unit and integration tests.
- Documentation for setup and API usage.
"""
}


VALID_ROLES = {
    "developer",
    "tester",
    "documentation",
    "reviewer",
}


def analyze_plan(plan):
    tasks = plan.get("tasks", [])

    ids = [
        task.get("id")
        for task in tasks
    ]

    id_set = set(ids)

    roles = Counter(
        task.get("role")
        for task in tasks
    )

    invalid_roles = [
        task
        for task in tasks
        if task.get("role") not in VALID_ROLES
    ]

    invalid_dependencies = []

    for task in tasks:
        for dep in task.get("depends_on", []):
            if dep not in id_set:
                invalid_dependencies.append(
                    (task.get("id"), dep)
                )

    duplicate_ids = (
        len(ids) != len(id_set)
    )

    ordering_errors = []

    id_positions = {
        task_id: i
        for i, task_id in enumerate(ids)
    }

    for task in tasks:
        current = task.get("id")

        for dep in task.get(
            "depends_on",
            [],
        ):
            if (
                dep in id_positions
                and current in id_positions
                and id_positions[dep]
                >= id_positions[current]
            ):
                ordering_errors.append(
                    (current, dep)
                )

    review_tasks = [
        task
        for task in tasks
        if task.get("role")
        == "reviewer"
    ]

    final_reviews = [
        task
        for task in review_tasks
        if "final" in task.get(
            "description",
            "",
        ).lower()
    ]

    roots = [
        task
        for task in tasks
        if not task.get(
            "depends_on",
            []
        )
    ]

    max_depth = compute_depth(
        tasks
    )

    return {
        "task_count": len(tasks),
        "roles": dict(roles),
        "duplicate_ids": duplicate_ids,
        "invalid_roles":
            len(invalid_roles),
        "invalid_dependencies":
            len(invalid_dependencies),
        "ordering_errors":
            len(ordering_errors),
        "root_tasks":
            len(roots),
        "review_tasks":
            len(review_tasks),
        "final_reviews":
            len(final_reviews),
        "max_depth":
            max_depth,
    }


def compute_depth(tasks):
    by_id = {
        task["id"]: task
        for task in tasks
        if "id" in task
    }

    memo = {}

    def depth(task_id, visiting=None):
        if task_id in memo:
            return memo[task_id]

        if visiting is None:
            visiting = set()

        if task_id in visiting:
            return 999

        visiting.add(task_id)

        task = by_id.get(task_id)

        if not task:
            return 0

        deps = task.get(
            "depends_on",
            []
        )

        if not deps:
            result = 1
        else:
            result = 1 + max(
                depth(
                    dep,
                    visiting.copy(),
                )
                for dep in deps
                if dep in by_id
            )

        memo[task_id] = result
        return result

    if not by_id:
        return 0

    return max(
        depth(task_id)
        for task_id in by_id
    )


def print_metrics(
    name,
    metrics,
):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    for key, value in (
        metrics.items()
    ):
        print(
            f"{key:24s}: {value}"
        )


def main():
    print(
        "Loading 7B planner model..."
    )

    llm = LocalLLM7B()
    planner = PlannerAgent(llm)

    results = {}

    for name, request in (
        PROJECTS.items()
    ):
        print(
            "\n\n"
            + "#" * 72
        )

        print(
            f"PLANNING {name}"
        )

        print(
            "#" * 72
        )

        try:
            plan = planner.plan(
                request
            )

            metrics = analyze_plan(
                plan
            )

            results[name] = {
                "plan": plan,
                "metrics": metrics,
            }

            print_metrics(
                name,
                metrics,
            )

            print(
                "\nGenerated plan:"
            )

            print(
                json.dumps(
                    plan,
                    indent=2,
                )
            )

        except Exception as exc:
            print(
                f"FAILED: {exc}"
            )

            results[name] = {
                "error": str(exc)
            }

    with open(
        "results/"
        "planner_benchmark.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nSaved:"
        " results/"
        "planner_benchmark.json"
    )


if __name__ == "__main__":
    main()
