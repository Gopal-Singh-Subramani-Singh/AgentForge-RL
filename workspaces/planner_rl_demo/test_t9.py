from app import TodoManager

def test_retrieve_existing_todo():
    manager = TodoManager()

    todo_id = manager.add_todo("A", "first")
    todo = manager.get_todo_by_id(todo_id)

    assert todo is not None
    assert todo.id == todo_id