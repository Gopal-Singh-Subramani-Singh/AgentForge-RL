from integration.planned_project import PlannedProject


PLAN = {
    "project": "Demo",
    "tasks": [
        {
            "id": "T1",
            "role": "developer",
            "description": "Implement feature",
            "depends_on": [],
        },
        {
            "id": "T2",
            "role": "tester",
            "description": "Test feature",
            "depends_on": ["T1"],
        },
        {
            "id": "T3",
            "role": "documentation",
            "description": "Document feature",
            "depends_on": ["T1"],
        },
        {
            "id": "T4",
            "role": "reviewer",
            "description": "Final review",
            "depends_on": ["T2", "T3"],
        },
    ],
}


project = PlannedProject(PLAN)

print("Initial:")
print(project.summary())
print(project.observation_vector())

assert [
    t["id"]
    for t in project.get_ready_tasks()
] == ["T1"]

task = project.pop_ready_task(
    "developer"
)

print("\nExecuting:")
print(task)

project.mark_completed("T1")
project.advance_step()

print("\nAfter T1:")
print(project.summary())
print(project.observation_vector())

ready_ids = {
    t["id"]
    for t in project.get_ready_tasks()
}

assert ready_ids == {
    "T2",
    "T3",
}

project.mark_completed("T2")
project.mark_completed("T3")
project.advance_step()

print("\nAfter testing/docs:")
print(project.summary())

assert [
    t["id"]
    for t in project.get_ready_tasks("reviewer")
] == ["T4"]

project.mark_completed("T4")

print("\nFinal:")
print(project.summary())

assert project.is_complete()

print("\nPlannedProject smoke test PASSED")
