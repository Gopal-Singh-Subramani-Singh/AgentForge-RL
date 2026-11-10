# Todo Management Module

## Description

This is a simple Todo management module implemented in Python. It allows you to manage your tasks by adding, retrieving, and removing them. The module provides functionalities to mark tasks as completed or incomplete.

## Features

1. **Add a Todo Item**: Adds a new task with a title and description.
2. **Get All Todos**: Retrieves all tasks.
3. **Get Completed Todos**: Retrieves all completed tasks.
4. **Get Incompleted Todos**: Retrieves all incompleted tasks.
5. **Get Todo by ID**: Retrieves a specific task by its unique identifier.
6. **Remove a Todo**: Removes a task by its unique identifier.

## Installation Instructions

To use this module, ensure you have Python installed on your system. No additional dependencies are required.

## How to Run the Application

Since this is a library, you can import it into your Python script or interactive environment. There is no executable script provided.

```python
from todo_manager import TodoManager

# Create a new instance of TodoManager
todo_manager = TodoManager()

# Add a new todo item
todo_manager.add_todo("Buy groceries", "Milk, eggs, bread")

# Get all todos
all_todos = todo_manager.get_all_todos()
print(all_todos)

# Mark a todo as completed
todo_to_complete = next(iter(all_todos.values()))
todo_to_complete.mark_as_completed()

# Get completed todos
completed_todos = todo_manager.get_completed_todos()
print(completed_todos)

# Remove a todo
todo_manager.remove_todo(todo_to_complete.id)

# Get remaining todos
remaining_todos = todo_manager.get_all_todos()
print(remaining_todos)
```

## API or Usage Documentation

### Class: `TodoItem`

#### Methods

- `__init__(title: str, description: str, completed: bool = False)`
  - Initializes a new `TodoItem` object.
  
- `mark_as_completed()`
  - Marks the todo item as completed.
  
- `mark_as_incomplete()`
  - Marks the todo item as incomplete.

### Class: `TodoManager`

#### Methods

- `__init__()`
  - Initializes a new `TodoManager` object.
  
- `add_todo(title: str, description: str)`
  - Adds a new todo item with the given title and description.
  
- `get_all_todos() -> dict[uuid.UUID, TodoItem]`
  - Returns a dictionary containing all todo items.
  
- `get_completed_todos() -> list[TodoItem]`
  - Returns a list of completed todo items.
  
- `get_incompleted_todos() -> list[TodoItem]`
  - Returns a list of incompleted todo items.
  
- `get_todo_by_id(todo_id: uuid.UUID) -> TodoItem`
  - Returns the todo item with the specified ID.
  
- `remove_todo(todo_id: uuid.UUID)`
  - Removes the todo item with the specified ID.

## Example Requests

Here's an example of how you might use this module in a script:

```python
from todo_manager import TodoManager

# Create a new instance of TodoManager
todo_manager = TodoManager()

# Add a new todo item
todo_manager.add_todo("Buy groceries", "Milk, eggs, bread")

# Get all todos
all_todos = todo_manager.get_all_todos()
print(all_todos)

# Mark a todo as completed
todo_to_complete = next(iter(all_todos.values()))
todo_to_complete.mark_as_completed()

# Get completed todos
completed_todos = todo_manager.get_completed_todos()
print(completed_todos)

# Remove a todo
todo_manager.remove_todo(todo_to_complete.id)

# Get remaining todos
remaining_todos = todo_manager.get_all_todos()
print(remaining_todos)
```

This will output the state of the todo manager at each step, showing how tasks are added, marked as completed, and removed.