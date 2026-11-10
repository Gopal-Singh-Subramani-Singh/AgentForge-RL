from pydantic import BaseModel

class Todo(BaseModel):
    id: int = None
    title: str
    completed: bool = False

todos_db = {}