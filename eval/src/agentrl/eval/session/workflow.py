from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from datetime import datetime, UTC
from typing import Any, Optional, Sequence, TYPE_CHECKING

from httpx import HTTPStatusError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionFunctionToolParam

from .types import InteractResponse, RunResult, RunSpec, SampleStatus
from ..convert import (FunctionDefinition,
                       MessageRecord,
                       OpenAIChatCompletionFunctionDefinition,
                       OpenAIChatCompletionInputMessageRecord)
from ..utils import DataUrlUtil

if TYPE_CHECKING:
    from .controller import ControllerClient
    from ..client import BaseClient


class SampleWorkflow:

    def __init__(self,
                 *,
                 controller: ControllerClient,
                 models: Sequence[BaseClient],
                 spec: RunSpec,
                 controller_renew: bool = False,
                 session_started_callback: Optional[Callable[[int], Coroutine[Any, Any, None]]] = None,
                 start_sample: bool = False):
        self.logger = logging.getLogger(__name__)

        # inputs
        self.controller = controller
        self.models = models  # cross-sample models, pick one at random each turn
        self.spec = spec
        self.controller_renew = controller_renew
        self.session_started_callback = session_started_callback
        self.start_sample = start_sample

        # internal state
        self.started_at: Optional[datetime] = None
        self.session_id: Optional[int] = None
        self.messages: list[MessageRecord] = []
        self.tools: list[FunctionDefinition] = []

    async def __call__(self) -> RunResult:
        self.started_at = datetime.now(tz=UTC)

        try:
            self.logger.debug('starting task="%s" index=%s run=%s',
                             self.spec.task, self.spec.index, self.spec.run)
            result = await self._run()
        except Exception as e:
            self.logger.debug('unhandled error', exc_info=True)
            result = self._client_complete(SampleStatus.WORKFLOW_ERROR, str(e))
        finally:
            if self.session_id is not None and (not locals().get('result') or result.status.is_client_error()):
                await self.cancel()

        if result.status.is_client_error():
            self.logger.error('error in task="%s" index=%s run=%s: %s',
                              self.spec.task, self.spec.index, self.spec.run, result.result)
        else:
            self.logger.info('completed task="%s" index=%s run=%s status="%s" reward=%s',
                             result.task, result.index, result.run, result.status, result.reward)

        return result

    async def _run(self) -> RunResult:
        # start sample with retry
        try:
            max_tries = 10
            for i in range(max_tries):
                try:
                    self.session_id, response = await self.controller.start_sample(
                        task=self.spec.task,
                        index=self.spec.index,
                        custom_params=self.spec.custom_params
                    )
                    break
                except HTTPStatusError as e:
                    if i == max_tries - 1:
                        raise e
                    self.logger.warning('failed to start sample: [%s] %s (try %d/%d)',
                                        e.response.status_code, e.response.text, i + 1, max_tries)
                    await asyncio.sleep(10)
            else:
                raise RuntimeError('unreachable code reached')  # ide hint
        except Exception as e:
            self.logger.debug('failed to start sample', exc_info=True)
            return self._client_complete(SampleStatus.SERVER_ERROR, str(e))

        if self.session_started_callback is not None:
            await self.session_started_callback(self.session_id)
        self.logger.debug('started session=%d for task="%s" index=%s run=%s',
                          self.session_id, self.spec.task, self.spec.index, self.spec.run)

        if self.start_sample:
            # for start_sample only mode, we immediately complete the session with a stub response
            await self.cancel()

            stub_response = response.model_copy()
            stub_response.finish = True
            stub_response.status = SampleStatus.COMPLETED
            stub_response.reward = 1.0
            return self._server_complete(stub_response)

        # interaction loop
        while True:
            # process response from last round (start_sample / interact)
            self._set_tools(response.tools)
            incremental_messages = self._set_messages(response.messages)
            self._log_messages('env messages: %s', incremental_messages)
            if response.finish:
                return self._server_complete(response)

            # cross_sampling: randomly pick one model from the list
            model = self.models[0]
            if len(self.models) > 1:
                model = random.choice(self.models)

            # controller_renew: create task to renew session if enabled
            renew_task: Optional[asyncio.Task] = None
            if self.controller_renew:
                renew_task = asyncio.create_task(self._schedule_renew())

            # call model client to get model messages
            try:
                messages = await model.query(
                    messages=self.messages,
                    tools=self.tools,
                    cache_key=self.spec.task_key()
                )
                assert messages, 'model returned empty messages'
            except Exception as e:
                self.logger.debug('failed to query model', exc_info=True)
                return self._client_complete(SampleStatus.MODEL_ERROR, str(e))
            finally:
                # controller_renew: ensure renew task is cancelled before continuing
                if renew_task is not None:
                    renew_task.cancel()
                    try:
                        await renew_task
                    except asyncio.CancelledError:
                        pass

            # add model messages to history
            self._log_messages('model messages: %s', messages)
            self.messages.extend(messages)

            # call controller to get next response from the environment
            try:
                response = await self.controller.interact(
                    session_id=self.session_id,
                    messages=MessageRecord.convert_all(messages, to='openai_chat_completion_input')
                )
            except Exception as e:
                self.logger.debug('failed to interact', exc_info=True)
                return self._client_complete(SampleStatus.SERVER_ERROR, str(e))

    async def cancel(self):
        if self.session_id is not None:
            self.logger.debug('cancelling session=%d', self.session_id)
            try:
                await self.controller.cancel(self.session_id)
            except Exception:
                pass  # we don't care about cancellation errors
            self.session_id = None

    async def _schedule_renew(self):
        assert self.session_id is not None
        while True:
            await asyncio.sleep(180)  # renew every 3 minutes
            await self.controller.renew(self.session_id)

    def _log_messages(self, fmt: str, messages: Sequence[MessageRecord]):
        if messages and self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug(fmt, DataUrlUtil.scrub(MessageRecord.dump_all(messages)))

    def _set_messages(self, messages: Sequence[ChatCompletionMessageParam]):
        # no empty check here; it's invalid for the messages to be empty.
        # the pydantic model should ensure this.

        # slices: since we are dealing with multiple message formats,
        # one message does not necessarily correspond to one message in another format.
        # so instead of by message, we organize messages produced by the same role into one slice.
        history: list[MessageRecord] = self.messages or []
        len_before = len(history)
        current_slice: list[ChatCompletionMessageParam] = []
        for message in messages:
            # if system prompt is present, we treat this as a full history.
            # the system message starts a new history of interaction.
            # push the system message into the first slice (a separate slice).
            if message['role'] == 'system' or message['role'] == 'developer':
                history = [OpenAIChatCompletionInputMessageRecord([message])]
                len_before = 0
                current_slice = []

            # the assistant message marks a separate slice.
            elif message['role'] == 'assistant' or message['role'] == 'tool':
                if current_slice:
                    history.append(OpenAIChatCompletionInputMessageRecord(current_slice))
                history.append(OpenAIChatCompletionInputMessageRecord([message]))
                current_slice = []

            # for user or tool messages, we group them together if multiple occurs in one turn.
            else:
                current_slice.append(message)
        if current_slice:
            history.append(OpenAIChatCompletionInputMessageRecord(current_slice))

        self.messages = history

        # return incremental messages for logging
        return self.messages[len_before:]

    def _set_tools(self, tools: Sequence[ChatCompletionFunctionToolParam]):
        if tools:
            original_tools_dump = FunctionDefinition.dump_all(self.tools)
            self.tools = [OpenAIChatCompletionFunctionDefinition(tool) for tool in tools]
            new_tools_dump = FunctionDefinition.dump_all(self.tools)
            if original_tools_dump != new_tools_dump:
                self.logger.debug('tools: %s', DataUrlUtil.scrub(new_tools_dump))
        elif not self.tools:
            self.tools = []  # Ensure tools is at least an empty list

    def _server_complete(self, response: InteractResponse) -> RunResult:
        score = response.metrics.pop('score', None) if isinstance(response.metrics, dict) else None
        task_trace = response.metrics.pop('trace', None) if isinstance(response.metrics, dict) else None

        return RunResult(
            model=self.spec.model,
            session_id=self.session_id,
            run=self.spec.run,
            task=self.spec.task,
            index=self.spec.index,
            status=response.status,
            reward=response.reward,
            score=score,
            metrics=response.metrics,
            result=response.result,
            task_trace=task_trace,
            raw_trace=MessageRecord.dump_all(self.messages),
            time_start=self.started_at,
            time_end=datetime.now(tz=UTC)
        )

    def _client_complete(self, status: SampleStatus, message: str) -> RunResult:
        return RunResult(
            model=self.spec.model,
            session_id=self.session_id,
            run=self.spec.run,
            task=self.spec.task,
            index=self.spec.index,
            status=status,
            reward=None,
            score=None,
            metrics=None,
            result=message,
            task_trace=None,
            raw_trace=MessageRecord.dump_all(self.messages),
            time_start=self.started_at,
            time_end=datetime.now(tz=UTC)
        )
