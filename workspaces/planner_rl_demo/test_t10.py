from app import TodoManager

def test_list_todos():
    manager = TodoManager()

    id1 = manager.add_todo("A", "first")
    id2 = manager.add_todo("B", "second")

    todos = manager.list_todos()
    ids = {todo.id for todo in todos}

    assert id1 in ids
    assert id2 in ids