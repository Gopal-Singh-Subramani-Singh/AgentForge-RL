from app import TodoManager

def test_full_todo_workflow():
    manager = TodoManager()

    todo_id = manager.add_todo(
        "Ship project",
        "Finish RL demo",
    )

    todo = manager.get_todo_by_id(todo_id)
    assert todo.id == todo_id

    all_todos = manager.list_todos()
    assert any(t.id == todo_id for t in all_todos)

    manager.mark_as_completed(todo_id)

    todo = manager.get_todo_by_id(todo_id)
    assert todo.completed is True

    manager.remove_todo(todo_id)

    remaining_ids = {
        todo.id
        for todo in manager.list_todos()
    }

    assert todo_id not in remaining_ids