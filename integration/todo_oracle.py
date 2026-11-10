ORACLE_TESTS = {
    "T8": """
import uuid
from app import TodoManager

def test_create_todo_returns_unique_ids():
    manager = TodoManager()

    id1 = manager.add_todo("A", "first")
    id2 = manager.add_todo("B", "second")

    assert isinstance(id1, uuid.UUID)
    assert isinstance(id2, uuid.UUID)
    assert id1 != id2
""",

    "T9": """
from app import TodoManager

def test_retrieve_existing_todo():
    manager = TodoManager()

    todo_id = manager.add_todo("A", "first")
    todo = manager.get_todo_by_id(todo_id)

    assert todo is not None
    assert todo.id == todo_id
""",

    "T10": """
from app import TodoManager

def test_list_todos():
    manager = TodoManager()

    id1 = manager.add_todo("A", "first")
    id2 = manager.add_todo("B", "second")

    todos = manager.list_todos()
    ids = {todo.id for todo in todos}

    assert id1 in ids
    assert id2 in ids
""",

    "T11": """
from app import TodoManager

def test_mark_complete():
    manager = TodoManager()

    todo_id = manager.add_todo("A", "first")

    manager.mark_as_completed(todo_id)

    todo = manager.get_todo_by_id(todo_id)

    assert todo.completed is True
""",

    "T12": """
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
""",

    "T13": """
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
""",
}
