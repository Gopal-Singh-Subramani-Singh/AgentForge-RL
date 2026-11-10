def create_todo(title):
    global next_id
    todo = {
        'id': next_id,
        'title': title,
        'completed': False
    }
    todos.append(todo)
    next_id += 1
    return todo

def get_todo(todo_id):
    for todo in todos:
        if todo['id'] == todo_id:
            return todo
    return None

def list_todos():
    return todos

def complete_todo(todo_id):
    for todo in todos:
        if todo['id'] == todo_id:
            todo['completed'] = True
            return todo
    return None

def delete_todo(todo_id):
    global next_id
    for i, todo in enumerate(todos):
        if todo['id'] == todo_id:
            del todos[i]
            next_id -= 1
            return True
    return False

todos = []
next_id = 1