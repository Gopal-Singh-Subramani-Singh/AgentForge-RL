# Todo Management API

## Description

This project provides a simple REST API for managing Todo items using FastAPI. The API allows you to create, retrieve, update, and delete Todo items.

## Features

- **POST /todos**: Create a new Todo item.
- **GET /todos**: Retrieve all Todo items.
- **GET /todos/{id}**: Retrieve a specific Todo item by ID.
- **PUT /todos/{id}**: Update a specific Todo item by ID.
- **DELETE /todos/{id}**: Delete a specific Todo item by ID.

## Installation Instructions

To install the required dependencies, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/todo-api.git
   cd todo-api
   ```

2. Install the dependencies using pip:
   ```sh
   pip install fastapi uvicorn
   ```

## How to Run the Application

To start the application, use the following command:

```sh
uvicorn app:app --reload
```

This will start the FastAPI server on `http://127.0.0.1:8000`.

## API Documentation

### Creating a Todo Item

**Endpoint**: `POST /todos`

**Request Body**:
```json
{
  "title": "Task Title"
}
```

**Response**:
```json
{
  "id": 1,
  "title": "Task Title",
  "completed": false
}
```

### Retrieving All Todos

**Endpoint**: `GET /todos`

**Response**:
```json
[
  {
    "id": 1,
    "title": "Task Title",
    "completed": false
  }
]
```

### Retrieving a Specific Todo

**Endpoint**: `GET /todos/{id}`

**Response**:
```json
{
  "id": 1,
  "title": "Task Title",
  "completed": false
}
```

### Updating a Specific Todo

**Endpoint**: `PUT /todos/{id}`

**Request Body**:
```json
{
  "title": "Updated Task Title",
  "completed": true
}
```

**Response**:
```json
{
  "id": 1,
  "title": "Updated Task Title",
  "completed": true
}
```

### Deleting a Specific Todo

**Endpoint**: `DELETE /todos/{id}`

**Response**:
```json
{
  "message": "Todo deleted"
}
```

## Example Requests

### Creating a Todo

```sh
curl -X POST "http://127.0.0.1:8000/todos" -H "Content-Type: application/json" -d '{"title": "Buy groceries"}'
```

### Retrieving All Todos

```sh
curl -X GET "http://127.0.0.1:8000/todos"
```

### Retrieving a Specific Todo

```sh
curl -X GET "http://127.0.0.1:8000/todos/1"
```

### Updating a Specific Todo

```sh
curl -X PUT "http://127.0.0.1:8000/todos/1" -H "Content-Type: application/json" -d '{"title": "Buy vegetables", "completed": true}'
```

### Deleting a Specific Todo

```sh
curl -X DELETE "http://127.0.0.1:8000/todos/1"
```

These endpoints provide a basic interface for managing Todo items using a RESTful API.