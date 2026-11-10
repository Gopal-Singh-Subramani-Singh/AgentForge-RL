from app import TodoManager

def test_delete_todo():
    manager = TodoManager()

    todo_id = manager.add_todo("A", "first")

    manager.remove_todo(todo_id)

    remaining_ids = {
        todo.id
        for todo in manager.list_todos()
    }

    assert todo_id not in remaining_ids