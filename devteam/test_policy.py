import ast


def _formatting_requested(
    project_request,
    task_description,
):
    combined = (
        project_request
        + "\n"
        + task_description
    ).lower()

    return any(
        phrase in combined
        for phrase in [
            "string representation",
            "__str__",
            "__repr__",
            "display format",
            "formatted output",
            "exact output",
            "print format",
        ]
    )


def _contains_str_repr(node):
    for child in ast.walk(
        node
    ):
        if isinstance(
            child,
            ast.Call,
        ):
            if (
                isinstance(
                    child.func,
                    ast.Name,
                )
                and child.func.id
                in {"str", "repr"}
            ):
                return True

    return False


class UnsafeTestVisitor(
    ast.NodeVisitor
):
    def __init__(
        self,
        allow_formatting=False,
    ):
        self.allow_formatting = (
            allow_formatting
        )

        self.bad = False
        self.reasons = []

    def visit_Assert(
        self,
        node,
    ):
        # Only flag string/repr use when it is
        # part of an assertion. This avoids
        # rejecting normal things such as:
        # str(tmp_path)
        if (
            not self.allow_formatting
            and _contains_str_repr(
                node.test
            )
        ):
            self.bad = True

            self.reasons.append(
                "unsupported exact "
                "string/repr behavior"
            )

        self.generic_visit(
            node
        )

    def visit_Attribute(
        self,
        node,
    ):
        if (
            node.attr.startswith("_")
            and not node.attr.startswith(
                "__"
            )
        ):
            self.bad = True

            self.reasons.append(
                "private implementation "
                "inspection"
            )

        self.generic_visit(
            node
        )


def sanitize_generated_test(
    project_request,
    task_description,
    source,
):
    try:
        tree = ast.parse(
            source
        )

    except SyntaxError:
        return (
            source,
            [],
        )

    allow_formatting = (
        _formatting_requested(
            project_request,
            task_description,
        )
    )

    kept = []
    removed = []

    for node in tree.body:
        is_test = (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name.startswith(
                "test_"
            )
        )

        if not is_test:
            kept.append(
                node
            )
            continue

        visitor = (
            UnsafeTestVisitor(
                allow_formatting=(
                    allow_formatting
                )
            )
        )

        visitor.visit(
            node
        )

        if visitor.bad:
            removed.append(
                {
                    "name":
                        node.name,
                    "reason":
                        ", ".join(
                            sorted(
                                set(
                                    visitor.reasons
                                )
                            )
                        ),
                }
            )

        else:
            kept.append(
                node
            )

    tree.body = kept

    try:
        cleaned = ast.unparse(
            tree
        )

    except Exception:
        return (
            source,
            [],
        )

    return (
        cleaned.strip()
        + "\n",
        removed,
    )


def validate_generated_test(
    project_request,
    task_description,
    source,
):
    try:
        tree = ast.parse(
            source
        )

    except SyntaxError as exc:
        return (
            False,
            "Generated test is not "
            f"valid Python: {exc}",
        )

    tests = [
        node
        for node in ast.walk(
            tree
        )
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name.startswith(
                "test_"
            )
        )
    ]

    if not tests:
        return (
            False,
            "Generated file contains "
            "no valid pytest test functions.",
        )

    allow_formatting = (
        _formatting_requested(
            project_request,
            task_description,
        )
    )

    visitor = UnsafeTestVisitor(
        allow_formatting=(
            allow_formatting
        )
    )

    visitor.visit(
        tree
    )

    if visitor.bad:
        return (
            False,
            "Generated test contains "
            "unsupported behavior: "
            + ", ".join(
                sorted(
                    set(
                        visitor.reasons
                    )
                )
            ),
        )

    return (
        True,
        "accepted",
    )


def developer_dependency_for_test(
    state,
    tester_task,
):
    dependencies = (
        tester_task.get(
            "depends_on",
            [],
        )
    )

    # Direct dependency first.
    for task_id in dependencies:
        task = state.tasks.get(
            task_id
        )

        if (
            task is not None
            and task.get("role")
            == "developer"
        ):
            return task

    # Then one parent level.
    for task_id in dependencies:
        parent = state.tasks.get(
            task_id
        )

        if parent is None:
            continue

        for parent_id in (
            parent.get(
                "depends_on",
                [],
            )
        ):
            task = (
                state.tasks.get(
                    parent_id
                )
            )

            if (
                task is not None
                and task.get(
                    "role"
                )
                == "developer"
            ):
                return task

    return None
