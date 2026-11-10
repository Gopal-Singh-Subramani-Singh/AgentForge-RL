import numpy as np


ROLES = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


class GenericProjectState:
    def __init__(
        self,
        plan,
        max_resources=2,
    ):
        self.project = plan.get(
            "project",
            "Software Project",
        )

        self.tasks = {}

        for task in plan.get(
            "tasks",
            [],
        ):
            item = dict(
                task
            )

            item.setdefault(
                "depends_on",
                [],
            )

            item["status"] = (
                "pending"
            )

            self.tasks[
                item["id"]
            ] = item

        self.step = 0

        self.max_resources = (
            max_resources
        )

        self.bugs = 0
        self.review_rejected = False

        self.refresh()

    def refresh(self):
        completed = {
            task_id
            for task_id, task
            in self.tasks.items()
            if task["status"]
            == "completed"
        }

        for task in (
            self.tasks.values()
        ):
            if task["status"] not in {
                "pending",
                "ready",
            }:
                continue

            deps = task.get(
                "depends_on",
                [],
            )

            if all(
                dep in completed
                for dep in deps
            ):
                task[
                    "status"
                ] = "ready"

    def ready_by_role(self):
        result = {
            role: []
            for role in ROLES
        }

        for task in (
            self.tasks.values()
        ):
            if (
                task["status"]
                != "ready"
            ):
                continue

            role = task.get(
                "role"
            )

            if role in result:
                result[
                    role
                ].append(
                    task
                )

        return result

    def pop_task(
        self,
        role,
    ):
        ready = (
            self.ready_by_role()[
                role
            ]
        )

        if not ready:
            return None

        task = ready[0]

        task["status"] = (
            "running"
        )

        return task

    def complete_task(
        self,
        task_id,
    ):
        self.tasks[
            task_id
        ]["status"] = (
            "completed"
        )

        self.refresh()

    def reopen(
        self,
        task_id,
    ):
        if task_id in self.tasks:
            self.tasks[
                task_id
            ]["status"] = (
                "ready"
            )

    def developer_tasks(self):
        return [
            task
            for task
            in self.tasks.values()
            if task.get("role")
            == "developer"
        ]

    def unfinished_developer(
        self,
    ):
        return [
            task
            for task
            in self.developer_tasks()
            if task["status"]
            != "completed"
        ]

    def completed_count(
        self,
    ):
        return sum(
            task["status"]
            == "completed"
            for task
            in self.tasks.values()
        )

    def total(self):
        return len(
            self.tasks
        )

    def complete(self):
        return (
            self.completed_count()
            == self.total()
        )

    def observation_vector(
        self,
    ):
        dev_remaining = sum(
            task["status"]
            != "completed"
            for task
            in self.tasks.values()
            if task.get("role")
            == "developer"
        )

        test_remaining = sum(
            task["status"]
            != "completed"
            for task
            in self.tasks.values()
            if task.get("role")
            == "tester"
        )

        docs_total = sum(
            task.get("role")
            == "documentation"
            for task
            in self.tasks.values()
        )

        docs_done = sum(
            task.get("role")
            == "documentation"
            and task["status"]
            == "completed"
            for task
            in self.tasks.values()
        )

        progress = (
            self.completed_count()
            / max(
                1,
                self.total(),
            )
        )

        docs_progress = (
            docs_done
            / max(
                1,
                docs_total,
            )
        )

        return np.array(
            [
                min(
                    dev_remaining / 10,
                    1,
                ),
                min(
                    test_remaining / 10,
                    1,
                ),
                min(
                    self.bugs / 5,
                    1,
                ),
                docs_progress,
                float(
                    docs_total > 0
                    and docs_done
                    == docs_total
                ),
                float(
                    self.review_rejected
                ),
                min(
                    self.max_resources / 4,
                    1,
                ),
                progress,
                1.0
                - min(
                    test_remaining / 10,
                    1,
                ),
                min(
                    self.step / 50,
                    1,
                ),
            ],
            dtype=np.float32,
        )

    def summary(self):
        ready = (
            self.ready_by_role()
        )

        return {
            "project":
                self.project,
            "step":
                self.step,
            "completed":
                self.completed_count(),
            "total":
                self.total(),
            "ready": {
                role: len(tasks)
                for role, tasks
                in ready.items()
            },
            "bugs":
                self.bugs,
            "complete":
                self.complete(),
        }
