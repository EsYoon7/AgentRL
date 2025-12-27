from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException
from google.protobuf import timestamp_pb2
from grpc.aio import insecure_channel
from starlette.responses import JSONResponse

from .pb import common_pb2, controller_v1_pb2, controller_v1_pb2_grpc

if TYPE_CHECKING:
    from typing import Any, Optional
    from grpc.aio import Channel, StreamStreamCall
    from .task_worker import TaskWorker


class GrpcTransport:

    def __init__(self, task_worker: TaskWorker):
        self.task_worker = task_worker
        self.logger = task_worker.logger
        assert self.task_worker.controller_address.startswith('grpc://'), \
            'controller address must start with grpc:// to use the grpc transport'
        self.controller_address = self.task_worker.controller_address.replace('grpc://', '').rstrip('/')
        self.message_queue: Optional[asyncio.Queue[controller_v1_pb2.WorkerStreamEnvelope]] = None
        self.running = False
        self._stream_task: Optional[asyncio.Task] = None

    async def start(self):
        self.running = True
        self.message_queue = asyncio.Queue()
        self._stream_task = asyncio.create_task(self._stream_loop())

    async def release(self):
        self.running = False
        if self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None

    async def _put_message(self, msg: controller_v1_pb2.WorkerStreamEnvelope):
        await self.message_queue.put(msg)

    async def _route_message(self, msg: controller_v1_pb2.WorkerStreamEnvelope):
        self.logger.debug('received gRPC message: %s', msg)
        if msg.type == controller_v1_pb2.WorkerStreamEnvelope.REQUEST and msg.worker_request:
            endpoint = msg.worker_request.endpoint
            try:
                if msg.worker_request.json:
                    data = json.loads(msg.worker_request.json.decode('utf-8'))
                else:
                    data = None
            except ValueError as e:
                self.logger.error('failed to decode gRPC request body', exc_info=e)
                data = None

            response_envelope = controller_v1_pb2.WorkerStreamEnvelope(
                id=msg.id,
                type=controller_v1_pb2.WorkerStreamEnvelope.RESPONSE,
                timestamp=timestamp_pb2.Timestamp(),
                worker_response=controller_v1_pb2.WorkerStreamEnvelope.WorkerResponse(
                    code=200,
                    message=None,
                    json=None
                )
            )
            response_envelope.timestamp.GetCurrentTime()

            try:
                method = getattr(self.task_worker, endpoint)
                sig = inspect.signature(method)
                params = list(sig.parameters.values())
                if len(params) == 0:
                    result = await method()
                elif len(params) == 1:
                    body_type = params[0].annotation
                    model_instance = body_type.model_validate(data) if isinstance(data, dict) else data
                    result = await method(model_instance)
                else:
                    raise ValueError('endpoint should have signature (self, parameters: Model)')

                if isinstance(result, HTTPException):
                    raise result

                if isinstance(result, JSONResponse):
                    response_envelope.worker_response.code = result.status_code
                    if result.body:
                        response_envelope.worker_response.json = result.body
                elif result is not None:
                    response_envelope.worker_response.json = json.dumps(result, ensure_ascii=False).encode('utf-8')

            except HTTPException as e:
                response_envelope.worker_response.code = e.status_code
                response_envelope.worker_response.message = str(e.detail)

            except Exception as e:
                self.logger.error('failed to handle gRPC request', exc_info=e)
                response_envelope.worker_response.code = 500
                response_envelope.worker_response.message = 'Internal Server Error'

            await self._put_message(response_envelope)

    async def _stream_recv_loop(self, stream: StreamStreamCall[controller_v1_pb2.WorkerStreamEnvelope, controller_v1_pb2.WorkerStreamEnvelope]):
        async for resp in stream:
            asyncio.create_task(self._route_message(resp))

    async def _stream_send_loop(self, stream: StreamStreamCall[controller_v1_pb2.WorkerStreamEnvelope, controller_v1_pb2.WorkerStreamEnvelope]):
        while True:
            try:
                msg = await self.message_queue.get()
                await stream.write(msg)
                self.logger.debug('sent gRPC message: %s', msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error('failed to send gRPC message', exc_info=e)
            finally:
                self.message_queue.task_done()

    async def _stream_loop(self):
        grpc_channel: Optional[Channel] = None
        grpc_stream: Optional[StreamStreamCall[controller_v1_pb2.WorkerStreamEnvelope, controller_v1_pb2.WorkerStreamEnvelope]] = None
        send_task: Optional[asyncio.Task] = None
        recv_task: Optional[asyncio.Task] = None

        while True:
            try:
                grpc_channel = insecure_channel(self.controller_address, options=[
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
                ])
                grpc_stub = controller_v1_pb2_grpc.ControllerStub(grpc_channel)

                grpc_stream = grpc_stub.WorkerStream()
                await grpc_stream.wait_for_connection()

                send_task = asyncio.create_task(self._stream_send_loop(grpc_stream))
                recv_task = asyncio.create_task(self._stream_recv_loop(grpc_stream))

                # close the stream whichever send or receive task finishes
                await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                self.logger.warning('gRPC stream closed unexpectedly, restarting...')
            except Exception as e:
                self.logger.error('gRPC stream error', exc_info=e)
            finally:
                if send_task:
                    try:
                        send_task.cancel()
                    except Exception:
                        pass
                    send_task = None
                if recv_task:
                    try:
                        recv_task.cancel()
                    except Exception:
                        pass
                    recv_task = None
                if grpc_stream:
                    try:
                        grpc_stream.cancel()
                    except Exception:
                        pass
                    grpc_stream = None
                if grpc_channel:
                    try:
                        await grpc_channel.close()
                    except Exception:
                        pass
                    grpc_channel = None

            await asyncio.sleep(1)  # wait before reconnecting

    async def send_heartbeat(self):
        # read information from the task worker and send a heartbeat
        envelope = controller_v1_pb2.WorkerStreamEnvelope(
            id=str(uuid4()),
            type=controller_v1_pb2.WorkerStreamEnvelope.HEARTBEAT,
            timestamp=timestamp_pb2.Timestamp(),
            receive_heartbeat_request=controller_v1_pb2.ReceiveHeartbeatRequest(
                id=self.task_worker.worker_id,
                name=self.task_worker.task.name,
                concurrency=self.task_worker.task.concurrency,
                indices=[
                    common_pb2.TaskIndex(
                        int_value=i if isinstance(i, int) else None,
                        string_value=i if isinstance(i, str) else None
                    )
                    for i in self.task_worker.task.get_indices()
                ]
            )
        )
        envelope.timestamp.GetCurrentTime()
        await self._put_message(envelope)
        self.logger.debug('sent gRPC message: %s', envelope)

    async def send_cancel_notice(self, session_id: int):
        # send a cancel notice to the controller
        envelope = controller_v1_pb2.WorkerStreamEnvelope(
            id=str(uuid4()),
            type=controller_v1_pb2.WorkerStreamEnvelope.REQUEST,
            timestamp=timestamp_pb2.Timestamp(),
            session_cancel_notice=controller_v1_pb2.SessionCancelNotice(
                session_id=session_id
            )
        )
        envelope.timestamp.GetCurrentTime()
        await self._put_message(envelope)
        self.logger.debug('sent gRPC request: %s', envelope)

    async def call_api(self, method: str, endpoint: str, data: Any = None, timeout: int = 30) -> Any:
        try:
            if data is not None and not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        except ValueError as e:
            self.logger.error('failed to encode gRPC body', exc_info=e)
            data = None
        request_envelope = controller_v1_pb2.WorkerStreamEnvelope(
            id=str(uuid4()),
            type=controller_v1_pb2.WorkerStreamEnvelope.REQUEST,
            timestamp=timestamp_pb2.Timestamp(),
            worker_request=controller_v1_pb2.WorkerStreamEnvelope.WorkerRequest(
                method=method,
                endpoint=endpoint,
                json=data
            )
        )
        request_envelope.timestamp.GetCurrentTime()
        await self._put_message(request_envelope)
        self.logger.debug('sent gRPC message: %s', request_envelope)

        # TODO: get response
        # it's not needed for now
