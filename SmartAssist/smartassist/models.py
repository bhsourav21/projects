from pydantic import BaseModel
from typing import Literal, Optional


class Task(BaseModel):
    """Represents a single step in a task list."""
    task_number: int
    task_description: str


class AgentResponse(BaseModel):
    """Discriminated wrapper so the model signals whether to render as text or a task list."""
    response_type: Literal["text", "task_list"]
    text: Optional[str] = None
    tasks: Optional[list[Task]] = None
