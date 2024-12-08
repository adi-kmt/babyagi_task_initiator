#!/usr/bin/env python
from dotenv import load_dotenv
from babyagi_task_initiator.schemas import (
    InputSchema, 
    TaskList, 
    Task, 
    TaskInitiatorPromptSchema, 
    TaskInitiatorAgentConfig
)
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.utils import get_logger
import json
import os
import instructor
from litellm import completion

load_dotenv()
logger = get_logger(__name__)

class TaskInitiatorAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment
        
        # Ensure agent_config is the correct type
        if isinstance(self.agent_deployment.agent_config, dict):
            self.agent_deployment.agent_config = TaskInitiatorAgentConfig(**self.agent_deployment.agent_config)
        
        # Set up the instructor-patched client
        self.client = instructor.patch(
            completion,
            mode=instructor.Mode.JSON
        )

    def generate_tasks(self, inputs: InputSchema) -> str:
        # Prepare user prompt
        user_prompt = self.agent_deployment.agent_config.user_message_template.replace(
            "{{objective}}", 
            inputs.tool_input_data.objective
        )
        
        # Prepare context if available
        context = inputs.tool_input_data.context
        if context:
            user_prompt += f"\nContext: {context}"
        
        # Prepare messages
        messages = [
            {"role": "system", "content": json.dumps(self.agent_deployment.agent_config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]
        
        # Prepare LLM configuration
        llm_config = self.agent_deployment.agent_config.llm_config
        api_key = None if llm_config.client == "ollama" else ("EMPTY" if llm_config.client == "vllm" else os.getenv("OPENAI_API_KEY"))
        
        # Make LLM call
        response = completion(
            model=self.agent_deployment.agent_config.llm_config.model,
            messages=messages,
            temperature=self.agent_deployment.agent_config.llm_config.temperature,
            max_tokens=self.agent_deployment.agent_config.llm_config.max_tokens,
            api_base=self.agent_deployment.agent_config.llm_config.api_base,
            api_key=api_key
        )
        
        logger.info(f"Generated Tasks: {response}")
        return response.model_dump_json()

def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {agent_run.inputs.tool_input_data}")
    task_initiator_agent = TaskInitiatorAgent(agent_run.agent_deployment)
    method = getattr(task_initiator_agent, agent_run.inputs.tool_name, None)
    return method(agent_run.inputs)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments
    
    naptha = Naptha()
    
    # Load agent deployments
    agent_deployments = load_agent_deployments(
        "babyagi_task_initiator/configs/agent_deployments.json", 
        load_persona_data=False, 
        load_persona_schema=False
    )
    
    # Prepare input parameters
    input_params = InputSchema(
        tool_name="generate_tasks",
        tool_input_data=TaskInitiatorPromptSchema(
            objective="Write a blog post about the weather in London.",
            context="Focus on historical weather patterns between 1900 and 2000"
        )
    )
    
    # Create agent run input
    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )
    
    # Run the agent
    response = run(agent_run)
    logger.info(f"Final Response: {response}")