import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_create_todo(client):
    response = client.post(
        "/todos",
        json={"title": "Learn RL"},
    )

    assert response.status_code in (200, 201)

    data = response.json()

    assert data["title"] == "Learn RL"
    assert data["completed"] is False
    assert isinstance(data["id"], int)


def test_get_all_todos(client):
    response = client.get("/todos")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unique_ids(client):
    first = client.post(
        "/todos",
        json={"title": "Task A"},
    ).json()

    second = client.post(
        "/todos",
        json={"title": "Task B"},
    ).json()

    assert first["id"] != second["id"]


def test_get_todo(client):
    created = client.post(
        "/todos",
        json={"title": "Read paper"},
    ).json()

    response = client.get(
        f"/todos/{created['id']}"
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Read paper"


def test_missing_todo(client):
    response = client.get("/todos/999999")

    assert response.status_code == 404


def test_update_todo(client):
    created = client.post(
        "/todos",
        json={"title": "Old title"},
    ).json()

    response = client.put(
        f"/todos/{created['id']}",
        json={
            "title": "New title",
            "completed": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["title"] == "New title"
    assert data["completed"] is True


def test_delete_todo(client):
    created = client.post(
        "/todos",
        json={"title": "Delete me"},
    ).json()

    todo_id = created["id"]

    response = client.delete(
        f"/todos/{todo_id}"
    )

    assert response.status_code in (200, 204)

    response = client.get(
        f"/todos/{todo_id}"
    )

    assert response.status_code == 404


def test_empty_title_rejected(client):
    response = client.post(
        "/todos",
        json={"title": ""},
    )

    assert response.status_code in (400, 422)
