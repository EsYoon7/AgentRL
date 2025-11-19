import asyncio
import logging
from typing import Optional, Tuple, List

from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
from grpc.aio import Channel, insecure_channel
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from .pb import common_pb2, snapshots_v1_pb2, snapshots_v1_pb2_grpc
from .typings import SampleIndex


class SnapshotsClient:

    def __init__(self, server_address: str):
        self.logger = logging.getLogger(__name__)

        self.server_address = server_address
        self._store_path: Optional[str] = None

        self._init_lock = asyncio.Lock()
        self._channel: Optional[Channel] = None
        self._client: Optional[snapshots_v1_pb2_grpc.SnapshotsManagerStub] = None

    async def _get_client(self) -> snapshots_v1_pb2_grpc.SnapshotsManagerStub:
        async with self._init_lock:
            if self._channel is None:
                self._channel = insecure_channel(self.server_address)
            await asyncio.wait_for(self._channel.channel_ready(), timeout=10)

            if self._client is None:
                self._client = snapshots_v1_pb2_grpc.SnapshotsManagerStub(self._channel)

            if self._store_path is None:
                response: snapshots_v1_pb2.GetStorePathResponse = await self._client.GetStorePath(empty_pb2.Empty())
                self._store_path = response.root_path

        return self._client

    async def get_store_path(self) -> str:
        await self._get_client()
        return self._store_path

    async def create_snapshot(self,
                              task_type: Optional[str] = None,
                              task_name: Optional[str] = None,
                              task_index: Optional[SampleIndex] = None,
                              env_type: Optional[str] = None,
                              session_id: Optional[int] = None,
                              step: Optional[int] = None,
                              parent_id: Optional[str] = None,
                              expected_size: Optional[int] = None,
                              tags: Optional[List[str]] = None) -> Tuple[str, str]:
        client = await self._get_client()
        request_kwargs = dict(
            task_type=task_type,
            task_name=task_name,
            task_index=common_pb2.TaskIndex(
                int_value=task_index if isinstance(task_index, int) else None,
                string_value=task_index if isinstance(task_index, str) else None
            ) if task_index is not None else None,
            env_type=env_type,
            session_id=session_id,
            step=step,
            parent_id=parent_id,
            expected_size=expected_size,
        )
        if tags:
            request_kwargs['tags'] = tags
        request = snapshots_v1_pb2.CreateSnapshotRequest(**request_kwargs)
        response: snapshots_v1_pb2.CreateSnapshotResponse = await client.CreateSnapshot(request)
        return response.id, response.path

    async def mark_ready(self, snapshot_id: str):
        client = await self._get_client()
        request = snapshots_v1_pb2.MarkReadyRequest(id=snapshot_id)
        await client.MarkReady(request)

    async def list_snapshots(self,
                             task_type: Optional[str] = None,
                             task_name: Optional[str] = None,
                             task_index: Optional[SampleIndex] = None,
                             env_type: Optional[str] = None,
                             session_id: Optional[int] = None,
                             step: Optional[int] = None,
                             parent_id: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             limit: int = 100) -> List[dict]:
        client = await self._get_client()
        request_kwargs = dict(
            task_type=task_type,
            task_name=task_name,
            task_index=common_pb2.TaskIndex(
                int_value=task_index if isinstance(task_index, int) else None,
                string_value=task_index if isinstance(task_index, str) else None
            ) if task_index is not None else None,
            env_type=env_type,
            session_id=session_id,
            step=step,
            parent_id=parent_id,
            page_size=limit
        )
        if tags:
            request_kwargs['tags'] = tags
        request = snapshots_v1_pb2.ListSnapshotsRequest(**request_kwargs)
        response: snapshots_v1_pb2.ListSnapshotsResponse = await client.ListSnapshots(request)
        return list([MessageToDict(s) for s in response.snapshots])

    async def get_snapshot(self, snapshot_id: str) -> Optional[dict]:
        client = await self._get_client()
        request = snapshots_v1_pb2.GetSnapshotRequest(id=snapshot_id)
        try:
            response: snapshots_v1_pb2.GetSnapshotResponse = await client.GetSnapshot(request)
            return MessageToDict(response.snapshot)
        except Exception as e:
            self.logger.warning(f'failed to get snapshot: {e}')
            return None

    async def get_snapshot_path(self, snapshot_id: str) -> str:
        client = await self._get_client()
        request = snapshots_v1_pb2.GetSnapshotRequest(id=snapshot_id, require_path=True)
        response: snapshots_v1_pb2.GetSnapshotResponse = await client.GetSnapshot(request)
        return response.path

    async def delete_snapshot(self, snapshot_id: str):
        client = await self._get_client()
        request = snapshots_v1_pb2.DeleteSnapshotRequest(id=snapshot_id)
        await client.DeleteSnapshot(request)

    async def add_snapshot_tags(self, snapshot_id: str, tags: List[str]) -> dict:
        client = await self._get_client()
        request = snapshots_v1_pb2.AddSnapshotTagsRequest(id=snapshot_id, tags=tags)
        response: snapshots_v1_pb2.Snapshot = await client.AddSnapshotTags(request)
        return MessageToDict(response)

    async def remove_snapshot_tags(self, snapshot_id: str, tags: List[str]) -> dict:
        client = await self._get_client()
        request = snapshots_v1_pb2.RemoveSnapshotTagsRequest(id=snapshot_id, tags=tags)
        response: snapshots_v1_pb2.Snapshot = await client.RemoveSnapshotTags(request)
        return MessageToDict(response)

    async def set_snapshot_tags(self, snapshot_id: str, tags: List[str]) -> dict:
        client = await self._get_client()
        request = snapshots_v1_pb2.SetSnapshotTagsRequest(id=snapshot_id, tags=tags)
        response: snapshots_v1_pb2.Snapshot = await client.SetSnapshotTags(request)
        return MessageToDict(response)

    @staticmethod
    def tools() -> List[ChatCompletionToolParam]:
        return [
            ChatCompletionToolParam(
                type='function',
                function=FunctionDefinition(
                    name='snapshot.create',
                    description='Create a new snapshot saving the current state of the environment.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'continue': {
                                'type': 'boolean',
                                'description': 'Whether to continue execution after creating the snapshot',
                                'default': True
                            },
                            'tags': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'List of tags to associate with the created snapshot'
                            }
                        },
                        'additionalProperties': False
                    },
                    strict=True
                )
            ),
            ChatCompletionToolParam(
                type='function',
                function=FunctionDefinition(
                    name='snapshot.list',
                    description='List available snapshots with optional filtering parameters.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'env_type': {
                                'type': 'string',
                                'description': 'Filter by environment type'
                            },
                            'session_id': {
                                'type': 'integer',
                                'description': 'Filter by session ID'
                            },
                            'step': {
                                'type': 'integer',
                                'description': 'Filter by step number'
                            },
                            'parent_id': {
                                'type': 'string',
                                'description': 'Filter by parent snapshot ID'
                            },
                            'tags': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Filter by snapshots that contain all specified tags'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Maximum number of snapshots to return',
                                'default': 100
                            }
                        },
                        'additionalProperties': False
                    },
                    strict=True
                )
            ),
            ChatCompletionToolParam(
                type='function',
                function=FunctionDefinition(
                    name='snapshot.load',
                    description='Load a snapshot by its ID to restore the environment state.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'id': {
                                'type': 'string',
                                'description': 'The ID of the snapshot to load'
                            }
                        },
                        'required': ['id'],
                        'additionalProperties': False
                    },
                    strict=True
                )
            ),
            ChatCompletionToolParam(
                type='function',
                function=FunctionDefinition(
                    name='snapshot.delete',
                    description='Delete a snapshot by its ID to free up storage.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'id': {
                                'type': 'string',
                                'description': 'The ID of the snapshot to delete'
                            }
                        },
                        'required': ['id'],
                        'additionalProperties': False
                    },
                    strict=True
                )
            )
        ]
