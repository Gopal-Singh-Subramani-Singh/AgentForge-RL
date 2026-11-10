
import importlib
import app


def setup_function():
    importlib.reload(app)


def test_create_todo():
    todo = app.create_todo("Write code")

    assert todo["id"] == 1
    assert todo["title"] == "Write code"
    assert todo["completed"] is False


def test_unique_ids():
    a = app.create_todo("A")
    b = app.create_todo("B")

    assert a["id"] != b["id"]


def test_get_todo():
    todo = app.create_todo("Read")

    result = app.get_todo(todo["id"])

    assert result == todo


def test_missing_todo():
    assert app.get_todo(999) is None


def test_list_todos():
    app.create_todo("A")
    app.create_todo("B")

    result = app.list_todos()

    assert len(result) == 2


def test_complete_todo():
    todo = app.create_todo("Finish")

    result = app.complete_todo(
        todo["id"]
    )

    assert result["completed"] is True


def test_complete_missing():
    assert app.complete_todo(999) is None


def test_delete_todo():
    todo = app.create_todo("Delete")

    assert app.delete_todo(
        todo["id"]
    ) is True

    assert app.get_todo(
        todo["id"]
    ) is None


def test_delete_missing():
    assert app.delete_todo(999) is False
