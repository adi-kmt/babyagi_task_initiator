from pydantic import BaseModel, Field
from typing import List
from naptha_sdk.schemas import AgentConfig

class TaskInitiatorPromptSchema(BaseModel):
    """Schema for task initiator input data"""
    objective: str
    context: str = Field(default="", description="Optional context for task generation")

class InputSchema(BaseModel):
    """Input schema matching the task executor's structure"""
    tool_name: str = Field(default="generate_tasks", description="Name of the method to call")
    tool_input_data: TaskInitiatorPromptSchema

class TaskInitiatorAgentConfig(AgentConfig):
    """Agent configuration for task initiator"""
    user_message_template: str = Field(..., description="Template for user message")

class Task(BaseModel):
    """Class for defining a task to be performed."""
    name: str = Field(..., description="The name of the task to be performed.")
    description: str = Field(..., description="The description of the task to be performed.")
    done: bool = Field(default=False, description="The status of the task. True if the task is done, False otherwise.")
    result: str = Field(default="", description="The result of the task.")

class TaskList(BaseModel):
    """Class for defining a list of tasks."""
    list: List[Task] = Field(default_factory=list, description="A list of tasks to be performed.")