from fastapi import FastAPI, HTTPException
from models import Todo, todos_db

app = FastAPI()

@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
    if not todo.title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    todo.id = len(todos_db) + 1
    todos_db[todo.id] = todo
    return todo

@app.get("/todos", response_model=list[Todo])
def get_todos():
    return list(todos_db.values())

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    todo = todos_db.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated_todo: Todo):
    todo = todos_db.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    todo.title = updated_todo.title or todo.title
    todo.completed = updated_todo.completed if updated_todo.completed is not None else todo.completed
    return todo

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos_db[todo_id]
    return {"message": "Todo deleted"}