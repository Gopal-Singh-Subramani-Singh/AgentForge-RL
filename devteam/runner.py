import json
from pathlib import Path

from llm.model_7b import (
    LocalLLM7B,
)
from planner.planner import (
    PlannerAgent,
)

from devteam.events import (
    EventLogger,
)
from devteam.live_scheduler import (
    LiveMARLScheduler,
)
from devteam.project_state import (
    GenericProjectState,
)
from devteam.python_adapter import (
    PythonProjectAdapter,
)
from devteam.role_executor import (
    GenericRoleExecutor,
)
from devteam.test_generator import (
    generate_test,
)
from devteam.test_policy import (
    sanitize_generated_test,
    validate_generated_test,
    developer_dependency_for_test,
)


def extract_plan(value):
    if isinstance(
        value,
        dict,
    ):
        if "tasks" in value:
            return value

        if isinstance(
            value.get("plan"),
            dict,
        ):
            return value["plan"]

    if isinstance(
        value,
        str,
    ):
        text = value.strip()

        first = text.find("{")
        last = text.rfind("}")

        if (
            first != -1
            and last != -1
        ):
            return json.loads(
                text[
                    first:last + 1
                ]
            )

    raise RuntimeError(
        "Planner did not return "
        "a valid plan."
    )


def make_plan(
    planner,
    request,
):
    methods = [
        "plan",
        "create_plan",
        "generate_plan",
    ]

    for name in methods:
        method = getattr(
            planner,
            name,
            None,
        )

        if method is None:
            continue

        try:
            return extract_plan(
                method(
                    request
                )
            )

        except TypeError:
            continue

    raise RuntimeError(
        "Could not find a supported "
        "PlannerAgent planning method."
    )


def write_result_files(
    adapter,
    result,
):
    files = result.get(
        "files",
        {},
    )

    if not isinstance(
        files,
        dict,
    ):
        return

    for path, content in (
        files.items()
    ):
        if not isinstance(
            content,
            str,
        ):
            continue

        adapter.write_file(
            path,
            content,
        )


def failing_test_task_ids(
    output,
):
    """
    Extract planner Tester task IDs from pytest
    filenames such as tests/test_t7.py.
    """

    import re

    ids = []

    for match in re.findall(
        r"test_(t[0-9]+)\\.py",
        output.lower(),
    ):
        task_id = match.upper()

        if task_id not in ids:
            ids.append(task_id)

    return ids


def choose_repair_task(
    state,
):
    completed = [
        task
        for task
        in state.developer_tasks()
        if task["status"]
        == "completed"
    ]

    if completed:
        return completed[-1]

    unfinished = (
        state.unfinished_developer()
    )

    if unfinished:
        return unfinished[0]

    return None


class DevTeamRunner:
    def __init__(
        self,
        request,
        workspace,
        resources=2,
        max_steps=40,
    ):
        self.request = request

        self.workspace = Path(
            workspace
        )

        self.resources = (
            resources
        )

        self.max_steps = (
            max_steps
        )

        self.events = (
            EventLogger(
                "results/live/"
                "events.jsonl"
            )
        )

        self.adapter = (
            PythonProjectAdapter(
                self.workspace
            )
        )

        self.llm = (
            LocalLLM7B()
        )

        self.planner = (
            PlannerAgent(
                self.llm
            )
        )

        self.roles = (
            GenericRoleExecutor(
                self.llm
            )
        )

        self.scheduler = (
            LiveMARLScheduler(
                events=self.events,
                max_resources=(
                    resources
                ),
            )
        )

        self.pending_repair = (
            None
        )

        self.test_attempts = {}
        self.last_test_failure = {}

    def run(self):
        self.events.emit(
            "project_start",
            request=self.request,
        )

        plan = make_plan(
            self.planner,
            self.request,
        )

        print(
            json.dumps(
                plan,
                indent=2,
            )
        )

        state = (
            GenericProjectState(
                plan,
                max_resources=(
                    self.resources
                ),
            )
        )

        self.events.emit(
            "plan",
            project=state.project,
            tasks=list(
                state.tasks.values()
            ),
        )

        try:
            for step in range(
                self.max_steps
            ):
                state.step = step
                state.refresh()

                summary = (
                    state.summary()
                )

                print()
                print("=" * 72)
                print(
                    "STEP",
                    step,
                )
                print("=" * 72)
                print(
                    summary
                )

                self.events.emit(
                    "step",
                    step=step,
                    completed=summary[
                        "completed"
                    ],
                    total=summary[
                        "total"
                    ],
                    resources=(
                        self.resources
                    ),
                    bugs=state.bugs,
                )

                if state.complete():
                    self.events.emit(
                        "complete",
                        step=step,
                    )

                    print(
                        "PROJECT COMPLETE"
                    )

                    return True

                observation = (
                    state
                    .observation_vector()
                )

                decision = (
                    self.scheduler
                    .decide(
                        observation
                    )
                )

                allocated = (
                    decision.get(
                        "allocated",
                        [],
                    )
                )

                ready = (
                    state.ready_by_role()
                )

                # Only execute roles that
                # actually have ready work.
                allocated = [
                    role
                    for role
                    in allocated
                    if ready.get(role)
                ]

                # -----------------------------------------
                # Safety fallback:
                # PPO remains primary scheduler, but an
                # out-of-distribution state must not allow
                # indefinite zero-progress deadlock.
                # -----------------------------------------
                if not allocated:
                    priority = [
                        "developer",
                        "tester",
                        "documentation",
                        "reviewer",
                    ]

                    fallback = next(
                        (
                            role
                            for role
                            in priority
                            if ready.get(role)
                        ),
                        None,
                    )

                    if (
                        fallback
                        is not None
                    ):
                        allocated = [
                            fallback
                        ]

                        print(
                            "[SAFETY FALLBACK]",
                            "Allocating",
                            fallback,
                        )

                        self.events.emit(
                            "scheduler_fallback",
                            role=fallback,
                            reason=(
                                "MARL selected "
                                "no ready role"
                            ),
                        )

                for role in allocated:
                    task = (
                        state.pop_task(
                            role
                        )
                    )

                    if task is None:
                        continue

                    self.events.emit(
                        "task_start",
                        role=role,
                        task_id=(
                            task["id"]
                        ),
                        description=(
                            task[
                                "description"
                            ]
                        ),
                    )

                    print()
                    print(
                        f"[{role.upper()}]"
                    )
                    print(
                        task["id"],
                        task[
                            "description"
                        ],
                    )

                    try:
                        # =============================
                        # DEVELOPER
                        # =============================
                        if (
                            role
                            == "developer"
                        ):
                            failure = None

                            if (
                                self.pending_repair
                                is not None
                                and
                                self.pending_repair.get(
                                    "developer_task"
                                )
                                == task["id"]
                            ):
                                failure = (
                                    self.pending_repair[
                                        "output"
                                    ]
                                )

                            result = (
                                self.roles
                                .developer(
                                    self.request,
                                    task[
                                        "description"
                                    ],
                                    self.adapter
                                    .files(),
                                    failure=(
                                        failure
                                    ),
                                )
                            )

                            write_result_files(
                                self.adapter,
                                result,
                            )

                            # Development tasks can
                            # complete before tests exist.
                            # A repair will later be
                            # validated by the Tester.
                            state.complete_task(
                                task["id"]
                            )

                        # =============================
                        # TESTER
                        # =============================
                        elif (
                            role
                            == "tester"
                        ):
                            attempt = (
                                self.test_attempts
                                .get(
                                    task["id"],
                                    0,
                                )
                                + 1
                            )

                            self.test_attempts[
                                task["id"]
                            ] = attempt

                            other_test_tasks = [
                                other[
                                    "description"
                                ]
                                for other
                                in state.tasks
                                .values()
                                if (
                                    other.get(
                                        "role"
                                    )
                                    == "tester"
                                    and
                                    other["id"]
                                    != task["id"]
                                )
                            ]

                            filename, source = (
                                generate_test(
                                    self.llm,
                                    self.request,
                                    task["id"],
                                    task[
                                        "description"
                                    ],
                                    self.adapter
                                    .files(),
                                    other_test_tasks=(
                                        other_test_tasks
                                    ),
                                    attempt=(
                                        attempt
                                    ),
                                    previous_failure=(
                                        self.last_test_failure
                                        .get(
                                            task[
                                                "id"
                                            ]
                                        )
                                    ),
                                )
                            )

                            if not source.strip():
                                print(
                                    "[TEST REJECTED]",
                                    "Tester produced "
                                    "no test code."
                                )

                                task[
                                    "status"
                                ] = "ready"

                                continue

                            (
                                source,
                                removed_tests,
                            ) = (
                                sanitize_generated_test(
                                    self.request,
                                    task[
                                        "description"
                                    ],
                                    source,
                                )
                            )

                            for removed in (
                                removed_tests
                            ):
                                print(
                                    "[TEST SANITIZED]",
                                    removed["name"],
                                    "-",
                                    removed[
                                        "reason"
                                    ],
                                )

                                self.events.emit(
                                    "test_sanitized",
                                    task_id=(
                                        task["id"]
                                    ),
                                    test=(
                                        removed[
                                            "name"
                                        ]
                                    ),
                                    reason=(
                                        removed[
                                            "reason"
                                        ]
                                    ),
                                )

                            (
                                valid,
                                rejection,
                            ) = (
                                validate_generated_test(
                                    self.request,
                                    task[
                                        "description"
                                    ],
                                    source,
                                )
                            )

                            if not valid:
                                print(
                                    "[TEST REJECTED]",
                                    rejection,
                                )

                                self.events.emit(
                                    "test_rejected",
                                    task_id=(
                                        task["id"]
                                    ),
                                    reason=(
                                        rejection
                                    ),
                                )

                                self.last_test_failure[
                                    task["id"]
                                ] = rejection

                                task[
                                    "status"
                                ] = "ready"

                                continue

                            self.adapter.write_file(
                                filename,
                                source,
                            )

                            quality = (
                                self.adapter
                                .quality_gate(
                                    test_file=filename
                                )
                            )

                            print(
                                quality[
                                    "output"
                                ]
                            )

                            self.events.emit(
                                "quality",
                                passed=quality[
                                    "passed"
                                ],
                                stage=quality[
                                    "stage"
                                ],
                                output=quality[
                                    "output"
                                ],
                            )

                            if quality[
                                "passed"
                            ]:
                                state.complete_task(
                                    task["id"]
                                )

                                self.test_attempts[
                                    task["id"]
                                ] = 0

                                self.last_test_failure.pop(
                                    task["id"],
                                    None,
                                )

                                if (
                                    self.pending_repair
                                    is not None
                                    and
                                    self.pending_repair
                                    .get(
                                        "tester_task"
                                    )
                                    == task["id"]
                                ):
                                    state.bugs = max(
                                        0,
                                        state.bugs - 1,
                                    )

                                    self.pending_repair = (
                                        None
                                    )

                            else:
                                task[
                                    "status"
                                ] = "ready"

                                self.last_test_failure[
                                    task["id"]
                                ] = quality[
                                    "output"
                                ]

                                repair_task = (
                                    developer_dependency_for_test(
                                        state,
                                        task,
                                    )
                                )

                                if (
                                    repair_task
                                    is None
                                ):
                                    repair_task = (
                                        choose_repair_task(
                                            state
                                        )
                                    )

                                if (
                                    repair_task
                                    is not None
                                ):
                                    repair_task[
                                        "status"
                                    ] = "ready"

                                    same_bug = (
                                        self.pending_repair
                                        is not None
                                        and
                                        self.pending_repair
                                        .get(
                                            "tester_task"
                                        )
                                        == task["id"]
                                    )

                                    if not same_bug:
                                        state.bugs += 1

                                    self.pending_repair = {
                                        "output":
                                            quality[
                                                "output"
                                            ],
                                        "tester_task":
                                            task[
                                                "id"
                                            ],
                                        "developer_task":
                                            repair_task[
                                                "id"
                                            ],
                                    }

                                    print(
                                        "Reopened Developer:",
                                        repair_task[
                                            "id"
                                        ],
                                    )

                        # =============================
                        # DOCUMENTATION
                        # =============================
                        elif (
                            role
                            == "documentation"
                        ):
                            result = (
                                self.roles
                                .documentation(
                                    self.request,
                                    task[
                                        "description"
                                    ],
                                    self.adapter
                                    .files(),
                                )
                            )

                            write_result_files(
                                self.adapter,
                                result,
                            )

                            state.complete_task(
                                task["id"]
                            )

                        # =============================
                        # REVIEWER
                        # =============================
                        elif (
                            role
                            == "reviewer"
                        ):
                            # Final regression gate runs ALL
                            # generated tests.
                            quality = (
                                self.adapter
                                .quality_gate()
                            )

                            print(
                                quality[
                                    "output"
                                ]
                            )

                            self.events.emit(
                                "quality",
                                passed=quality[
                                    "passed"
                                ],
                                stage=quality[
                                    "stage"
                                ],
                                output=quality[
                                    "output"
                                ],
                            )

                            if not quality[
                                "passed"
                            ]:
                                failing_ids = (
                                    failing_test_task_ids(
                                        quality[
                                            "output"
                                        ]
                                    )
                                )

                                reopened = []

                                for tester_id in (
                                    failing_ids
                                ):
                                    tester_task = (
                                        state.tasks.get(
                                            tester_id
                                        )
                                    )

                                    if tester_task is None:
                                        continue

                                    developer_task = (
                                        developer_dependency_for_test(
                                            state,
                                            tester_task,
                                        )
                                    )

                                    if (
                                        developer_task
                                        is not None
                                    ):
                                        developer_task[
                                            "status"
                                        ] = "ready"

                                        tester_task[
                                            "status"
                                        ] = "ready"

                                        reopened.append(
                                            developer_task[
                                                "id"
                                            ]
                                        )

                                task[
                                    "status"
                                ] = "ready"

                                state.review_rejected = True

                                print(
                                    "Reviewer regression "
                                    "failure. Reopened:",
                                    reopened,
                                )

                                continue

                            review = (
                                self.roles
                                .review(
                                    self.request,
                                    self.adapter
                                    .files(),
                                    quality[
                                        "output"
                                    ],
                                )
                            )

                            print(
                                "Review:",
                                review,
                            )

                            approved = bool(
                                review.get(
                                    "approved",
                                    False,
                                )
                            )

                            if approved:
                                state.complete_task(
                                    task["id"]
                                )

                                state.review_rejected = (
                                    False
                                )

                            else:
                                state.review_rejected = (
                                    True
                                )

                                task[
                                    "status"
                                ] = "ready"

                    finally:
                        self.events.emit(
                            "task_end",
                            role=role,
                            task_id=(
                                task["id"]
                            ),
                        )

            print(
                "MAX STEPS REACHED"
            )

            self.events.emit(
                "max_steps",
                step=(
                    self.max_steps
                ),
            )

            return False

        finally:
            self.scheduler.close()
