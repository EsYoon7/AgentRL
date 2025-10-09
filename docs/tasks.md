# Tasks

## Table of Contents

- [Integrated Tasks](#integrated-tasks)
  - [AgentBench FC](#agentbench-fc)
  - [MobileRL (Android)](#mobilerl-android)
- [Extending Tasks](#extending-tasks)
  - [The Environment Framework](#the-environment-framework)
  - [Common Pitfalls](#common-pitfalls)

## Integrated Tasks

We provide first-party integration for the following tasks into the environment framework:

### AgentBench FC

We have refactored the original [AgentBench](https://github.com/THUDM/AgentBench),
supporting a function-calling style prompt and containerized deployment.

Available in the [agentbench_fc](https://github.com/THUDM/AgentBench/tree/agentbench_fc) branch of the original repository.

### MobileRL (Android)

This task integrates the [Android Lab](https://github.com/THUDM/Android-Lab) test set
and [Android World](https://github.com/google-research/android_world) test set.

Available in the [MobileRL](https://github.com/THUDM/MobileRL/tree/main/inference) repository.

## Extending Tasks

To integrate your own task into the environment framework, the core is to implement a subclass of `Task`.
Here's an overview of the APIs of a task that should be implemented:

```python
class Task:

    def __init__(self,
                 name: str,
                 concurrency: int = 16,
                 tools: Optional[list] = None,
                 full_async: bool = False,
                 *args, **kwargs):
        """
        :param name: Name of the task, will be used as an identifier for the task and is displayed in the dashboard.
        :param concurrency: Max number of concurrent sessions that one worker can handle.
        :param tools: If the task uses function calling for interaction, specify the tools in OpenAI format here.
        :param full_async: If True, the task is considered fully asynchronous,
                           meaning that cancellation of a session will directly `cancel()` the coroutine.
                           This should only be set to True if the task does not call any blocking code,
                           even if through to_thread or run_in_executor.
        """

    def get_indices(self) -> List[SampleIndex]:
        """
        Return a list of indices for the task. Indices can be str or int.
        """

    def sync_start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """
        Synchronous version of `start_sample`, could use `session.sync_action()` instead of `await session.action()`.
        """

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """
        Start a sample with the given index and session.
        The default implementation is to call `sync_start_sample` in a thread pool executor.
        Must be implemented if `full_async` is set to True.
        """

    async def start_sample_custom(self, task: dict, session: Session) -> TaskSampleExecutionResult:
        """
        For tasks that would like to support custom tasks at runtime,
        include -1 in `get_indices` and override this method.
        """
```

The main entry point for your task would be `start_sample` or `sync_start_sample`,
which should contain the logic of initializing the environment, observations,
execution of the agent's response, and evaluating the agent's performance.

The `session` object provides several methods to interact with the agent:

- `session.inject(item)`: Inject an OpenAI-format message or reward into the session.
  All system prompts, user prompts and tool prompts should be provided to the agent through this method.
- `await session.action()` or `session.sync_action()`: Return all previously injected messages to the agent,
  and wait for the agent's response.
- `session.set_tools(tools)`: Set the list of tools that the agent can use for function calling.
  The method only works before the first call to `session.action()`, 
  and is intended to be used if each sample of the task may have different tools.
  Setting the `tools` attribute of the `Task` is preferred over this method.

When interaction is complete, a reward should be injected using `session.inject(RewardHistoryItem(reward))`.
The `RewardHistoryItem` can optionally contain additional metrics, please see its source for more details.

By default, only incremental messages are returned in each round of interaction.
It is possible to override this behavior, by calling `session.set_full_history(True)` before injecting messages,
and replacing `session.inject(message)` with `session.cover(messages)`.

### The Environment Framework

Many tasks need an external environment (e.g., DB server, OS container).
The environment framework splits this into two roles:

- `EnvironmentDelegation` (task-specific): Describes what kinds of environments you need and how to build them.
- `EnvironmentController` (driver): Implements how to allocate/manage those environments at runtime.

For tasks that wish to use the environment framework to manage their external environments,
a subclass of `EnvironmentDelegation` should be implemented.

The `EnvironmentDelegation` base class is commented in detail, you can refer to its source for more information.

Here's an example of how to include the environment controller in a task:

```yaml
# in the configuration file
default:
  parameters:
    env_driver: docker
    env_options:
      network_name: demo_default
      state_driver: redis
      state_options:
        connection:
          host: 172.17.0.1
```

```python
class DemoTask(Task):

    def __init__(self,
                 env_driver: str = 'docker',
                 env_options: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.env_delegation = DemoEnvironmentDelegation()
        self.env_controller = create_controller(env_driver, self.env_delegation, **env_options)
        self.env_controller_background_task = None

    async def start_sample(self, index: int, session: Session) -> TaskSampleExecutionResult:
        self.env_controller.loop = asyncio.get_running_loop()
        if not self.env_controller_background_task:
            self.env_controller_background_task = asyncio.create_task(self.env_controller.background_task())
            weakref.finalize(self, self.env_controller_background_task.cancel)

        # Create an environment for the session
        session_id, container_ids, urls = await self.env_controller.start_session('default')

        try:
            for _ in range(20):  # main interaction loop
                await self.env_controller.renew_session(session_id)
                await session.action()
        finally:
            # ensure the environment is cleaned up
            await self.env_controller.end_session(session_id)
```

Currently, only Docker is supported as an environment driver.
It is possible to support more backends by implementing the `EnvironmentController` interface.

### Common Pitfalls

- To handle a large number of concurrent sessions, the event loop must not be blocked under any conditions.
  If your task requires blocking code, you should use `sync_start_sample` instead.

- Cancellation of sessions is implemented by raising an `AgentCancelledException` when `session.action()` is called.
  If this exception is caught and ignored, the session will never successfully cancel and keep occupying concurrency.
