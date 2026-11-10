# FILE: app.py
import uuid

class TodoItem:
    def __init__(self, title: str, description: str, completed: bool = False):
        self.id = uuid.uuid4()
        self.title = title
        self.description = description
        self.completed = completed

    def mark_as_completed(self):
        self.completed = True

    def mark_as_incomplete(self):
        self.completed = False


class TodoManager:
    def __init__(self):
        self.todos = {}

    def add_todo(self, title: str, description: str) -> uuid.UUID:
        new_todo = TodoItem(title, description)
        self.todos[new_todo.id] = new_todo
        return new_todo.id

    def get_all_todos(self) -> dict[uuid.UUID, TodoItem]:
        return self.todos

    def get_completed_todos(self) -> list[TodoItem]:
        return [todo for todo in self.todos.values() if todo.completed]

    def get_incompleted_todos(self) -> list[TodoItem]:
        return [todo for todo in self.todos.values() if not todo.completed]

    def get_todo_by_id(self, todo_id: uuid.UUID) -> TodoItem | None:
        return self.todos.get(todo_id)

    def remove_todo(self, todo_id: uuid.UUID):
        if todo_id in self.todos:
            del self.todos[todo_id]

    def list_todos(self) -> list[TodoItem]:
        return list(self.todos.values())

    def mark_as_completed(self, todo_id: uuid.UUID):
        todo = self.todos.get(todo_id)
        if todo:
            todo.mark_as_completed()

    def mark_as_incomplete(self, todo_id: uuid.UUID):
        todo = self.todos.get(todo_id)
        if todo:
            todo.mark_as_incomplete()