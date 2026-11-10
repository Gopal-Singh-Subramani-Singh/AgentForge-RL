import uuid
from app import TodoManager

def test_create_todo_returns_unique_ids():
    manager = TodoManager()

    id1 = manager.add_todo("A", "first")
    id2 = manager.add_todo("B", "second")

    assert isinstance(id1, uuid.UUID)
    assert isinstance(id2, uuid.UUID)
    assert id1 != id2