import json
import logging
from typing import List

from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)

from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleIndex,
                                    SampleStatus,
                                    TaskSampleExecutionResult)

# simple tools for the agent to choose from
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "number",
                        "description": "The final result"
                    }
                },
                "required": ["result"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


# generate some simple arithmetic tasks
def generate_tasks(count: int):
    return [
        {
            'a': i,
            'b': i * 2,
            'op': '+' if i % 2 == 0 else '-',
            'ans': i + (i * 2) if i % 2 == 0 else i - (i * 2)
        }
        for i in range(1, count + 1)
    ]


class SimpleCalculatorTask(Task):

    def __init__(self,
                 max_rounds: int,
                 number_of_indices: int,
                 **kwargs):
        super().__init__(tools=TOOLS, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.max_rounds = max_rounds
        self.tasks = generate_tasks(number_of_indices)

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.tasks)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        # get the task
        task = self.tasks[index]
        self.logger.info(f'Starting task {index}: {task["a"]} {task["op"]} {task["b"]} (ans: {task["ans"]})')

        # inject system prompt
        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content='You are an agent specialized in tool-calling. '
                    'You can use the functions provided to you to perform calculations. '
                    'When you are done, you must use the "submit" function to submit the final answer. '
                    'Do not provide any text response outside of the function calls.'
        ))

        # inject user prompt
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=f"What's {task['a']} {task['op']} {task['b']}?"
        ))

        # the interaction loop with error handling
        try:
            for _ in range(self.max_rounds):

                # wait for agent response
                response = await session.action()
                tool_calls = response.messages[0].get('tool_calls') or []

                # assert agent called a tool
                if not tool_calls:
                    return TaskSampleExecutionResult(status=SampleStatus.AGENT_VALIDATION_FAILED)

                # handle execution of tools
                for tool_call in tool_calls:
                    self.logger.info(f'agent tool call: {tool_call}')

                    tool_call_id = tool_call.get('id')
                    tool_name = tool_call.get('function', {}).get('name')
                    try:
                        tool_args = json.loads(tool_call.get('function', {}).get('arguments'))
                    except ValueError:
                        tool_args = {}

                    if tool_name == 'add' or tool_name == 'subtract':
                        a = tool_args.get('a')
                        b = tool_args.get('b')
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                            result = a + b if tool_name == 'add' else a - b
                            session.inject(ChatCompletionToolMessageParam(
                                role='tool',
                                content=str(result),
                                tool_call_id=tool_call_id
                            ))
                        else:
                            session.inject(ChatCompletionToolMessageParam(
                                role='tool',
                                content='Invalid arguments. Both a and b must be numbers.',
                                tool_call_id=tool_call_id
                            ))

                    elif tool_name == 'submit':
                        result = tool_args.get('result')
                        reward = float(result == task['ans'])

                        # inject the reward
                        session.inject(RewardHistoryItem(reward=reward))
                        return TaskSampleExecutionResult(status=SampleStatus.COMPLETED)

                    else:
                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            content=f'Unknown function: {tool_name}',
                            tool_call_id=tool_call_id
                        ))

            # handle if max rounds reached without submission
            else:
                return TaskSampleExecutionResult(status=SampleStatus.TASK_LIMIT_REACHED)

        except AgentCancelledException:
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
        except Exception:
            self.logger.exception('error in task execution')
            return TaskSampleExecutionResult(status=SampleStatus.TASK_ERROR)
