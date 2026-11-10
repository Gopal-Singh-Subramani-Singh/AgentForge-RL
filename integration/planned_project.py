from collections import defaultdict


class PlannedProject:
    def __init__(self, plan):
        self.project_name = plan.get("project", "Unnamed Project")

        self.tasks = {}
        self.order = []

        for task in plan.get("tasks", []):
            task_id = task["id"]

            self.tasks[task_id] = {
                "id": task_id,
                "role": task["role"],
                "description": task["description"],
                "depends_on": list(task.get("depends_on", [])),
                "status": "pending",
            }

            self.order.append(task_id)

        self.step = 0
        self.bugs = 0
        self.review_rejected = False

        self._refresh_ready_tasks()

    def _dependencies_complete(self, task):
        return all(
            self.tasks[dep]["status"] == "completed"
            for dep in task["depends_on"]
        )

    def _refresh_ready_tasks(self):
        for task in self.tasks.values():
            if task["status"] != "pending":
                continue

            if self._dependencies_complete(task):
                task["status"] = "ready"

    def get_ready_tasks(self, role=None):
        self._refresh_ready_tasks()

        ready = []

        for task_id in self.order:
            task = self.tasks[task_id]

            if task["status"] != "ready":
                continue

            if role is not None and task["role"] != role:
                continue

            ready.append(task)

        return ready

    def pop_ready_task(self, role):
        ready = self.get_ready_tasks(role)

        if not ready:
            return None

        task = ready[0]
        task["status"] = "running"

        return task

    def mark_completed(self, task_id):
        if task_id not in self.tasks:
            raise KeyError(task_id)

        self.tasks[task_id]["status"] = "completed"

        if self.tasks[task_id]["role"] == "reviewer":
            self.review_rejected = False

        self._refresh_ready_tasks()

    def mark_failed(self, task_id):
        if task_id not in self.tasks:
            raise KeyError(task_id)

        task = self.tasks[task_id]

        task["status"] = "ready"

        if task["role"] == "developer":
            self.bugs += 1

        if task["role"] == "reviewer":
            self.review_rejected = True

    def add_bug(self):
        self.bugs += 1

    def resolve_bug(self):
        self.bugs = max(
            0,
            self.bugs - 1,
        )

    def advance_step(self):
        self.step += 1
        self._refresh_ready_tasks()

    def is_complete(self):
        return all(
            task["status"] == "completed"
            for task in self.tasks.values()
        )

    def completed_count(self):
        return sum(
            task["status"] == "completed"
            for task in self.tasks.values()
        )

    def role_counts(self):
        counts = defaultdict(int)

        for task in self.tasks.values():
            counts[task["role"]] += 1

        return dict(counts)

    def role_completed(self, role):
        return sum(
            task["role"] == role
            and task["status"] == "completed"
            for task in self.tasks.values()
        )

    def role_remaining(self, role):
        return sum(
            task["role"] == role
            and task["status"] != "completed"
            for task in self.tasks.values()
        )

    def ready_role_counts(self):
        counts = {
            "developer": 0,
            "tester": 0,
            "documentation": 0,
            "reviewer": 0,
        }

        for task in self.get_ready_tasks():
            counts[task["role"]] += 1

        return counts

    def get_scheduler_state(self, resources=2, max_steps=50):
        """
        Maps an arbitrary planner DAG into the 10-dimensional
        observation expected by the existing MARL policy.
        """

        total = max(
            1,
            len(self.tasks),
        )

        completed = self.completed_count()

        developer_remaining = self.role_remaining("developer")
        tester_ready = len(
            self.get_ready_tasks("tester")
        )

        docs_total = sum(
            task["role"] == "documentation"
            for task in self.tasks.values()
        )

        docs_completed = self.role_completed(
            "documentation"
        )

        if docs_total:
            docs_progress = docs_completed / docs_total
        else:
            docs_progress = 1.0

        docs_complete = (
            docs_total == 0
            or docs_completed == docs_total
        )

        dev_total = sum(
            task["role"] == "developer"
            for task in self.tasks.values()
        )

        dev_completed = self.role_completed(
            "developer"
        )

        if dev_total:
            development_progress = (
                dev_completed / dev_total
            )
        else:
            development_progress = 1.0

        test_total = sum(
            task["role"] == "tester"
            for task in self.tasks.values()
        )

        tests_completed = self.role_completed(
            "tester"
        )

        if test_total:
            test_progress = (
                tests_completed / test_total
            )
        else:
            test_progress = 1.0

        return {
            "dev_tasks": min(
                developer_remaining / max(dev_total, 1),
                1.0,
            ),

            "untested_work": min(
                tester_ready / max(test_total, 1),
                1.0,
            ),

            "bugs": min(
                self.bugs / 5.0,
                1.0,
            ),

            "docs_progress": docs_progress,

            "docs_complete": float(
                docs_complete
            ),

            "review_rejected": float(
                self.review_rejected
            ),

            "resources": min(
                resources / 4.0,
                1.0,
            ),

            "development_progress":
                development_progress,

            "test_progress":
                test_progress,

            "time": min(
                self.step / max_steps,
                1.0,
            ),
        }

    def observation_vector(self, resources=2, max_steps=50):
        state = self.get_scheduler_state(
            resources=resources,
            max_steps=max_steps,
        )

        return [
            state["dev_tasks"],
            state["untested_work"],
            state["bugs"],
            state["docs_progress"],
            state["docs_complete"],
            state["review_rejected"],
            state["resources"],
            state["development_progress"],
            state["test_progress"],
            state["time"],
        ]

    def summary(self):
        ready = self.ready_role_counts()

        return {
            "project": self.project_name,
            "step": self.step,
            "completed": self.completed_count(),
            "total": len(self.tasks),
            "ready": ready,
            "bugs": self.bugs,
            "review_rejected": self.review_rejected,
            "complete": self.is_complete(),
        }
