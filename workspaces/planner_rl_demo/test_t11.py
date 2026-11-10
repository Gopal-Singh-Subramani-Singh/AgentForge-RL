from app import TodoManager

def test_mark_complete():
    manager = TodoManager()

    todo_id = manager.add_todo("A", "first")

    manager.mark_as_completed(todo_id)

    todo = manager.get_todo_by_id(todo_id)

    assert todo.completed is True