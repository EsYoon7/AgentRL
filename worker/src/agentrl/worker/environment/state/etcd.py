from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

import grpc

from ._base import StateProvider

_PROTO_ENV = 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'
_prev_proto_impl = os.environ.get(_PROTO_ENV)
os.environ[_PROTO_ENV] = 'python'
from etcd3.etcdrpc import rpc_pb2, rpc_pb2_grpc
if _prev_proto_impl is None:
    del os.environ[_PROTO_ENV]
else:
    os.environ[_PROTO_ENV] = _prev_proto_impl


class EtcdStateProvider(StateProvider):
    """
    This is an experimental state provider that is not battle-tested. Use with caution.
    """

    def __init__(self, connection: Optional[Any] = None, prefix: str = ''):
        self.logger = logging.getLogger(__name__)
        if isinstance(connection, str):
            connection = {'endpoint': connection}
        connection = connection or {}
        self.prefix = f'{prefix}:' if prefix else ''
        host = connection.get('host', '127.0.0.1')
        port = connection.get('port', 2379)
        self._target = connection.get('endpoint') or f'{host}:{port}'
        self._tls = connection.get('tls', False)
        self._ca_cert = connection.get('ca_cert')
        self._cert = connection.get('cert')
        self._key = connection.get('key')
        self._user = connection.get('user')
        self._password = connection.get('password')
        self.session_expiry = int(connection.get('session_expiry', 600))

        self._channel_options = connection.get('grpc_options') or [
            ('grpc.enable_retries', 1),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.initial_reconnect_backoff_ms', 500),
            ('grpc.max_reconnect_backoff_ms', 5000),
        ]
        self._retryable_statuses = (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.INTERNAL,
        )

        self._metadata = self._build_metadata()
        self._channel: Optional[grpc.aio.Channel] = None
        self._kv_stub: Optional[rpc_pb2_grpc.KVStub] = None
        self._lease_stub: Optional[rpc_pb2_grpc.LeaseStub] = None
        self._keepalive_stream: Optional[grpc.aio.StreamStreamCall] = None
        self._keepalive_lock = asyncio.Lock()

        self._session_leases: Dict[str, int] = {}
        self._lock_leases: Dict[str, int] = {}
        self._lock_keys: Dict[Tuple[str, str], bytes] = {}
        self._default_lock_client = f'lock-{uuid.uuid4()}'

    def _build_metadata(self) -> List[Tuple[str, str]]:
        if self._user and self._password:
            token = base64.b64encode(f'{self._user}:{self._password}'.encode()).decode()
            return [('authorization', f'Basic {token}')]
        return []

    async def close(self):
        await self._reset_channel()

    async def _get_channel(self) -> grpc.aio.Channel:
        if self._channel is None:
            if self._tls:
                creds = grpc.ssl_channel_credentials(
                    root_certificates=self._read_file(self._ca_cert),
                    private_key=self._read_file(self._key),
                    certificate_chain=self._read_file(self._cert),
                )
                self._channel = grpc.aio.secure_channel(self._target, creds, options=self._channel_options)
            else:
                self._channel = grpc.aio.insecure_channel(self._target, options=self._channel_options)
        return self._channel

    async def _reset_channel(self):
        await self._reset_keepalive_stream()
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
        self._kv_stub = None
        self._lease_stub = None

    @staticmethod
    def _read_file(path: Optional[str]) -> Optional[bytes]:
        if not path:
            return None
        with open(path, 'rb') as fp:
            return fp.read()

    async def _get_kv(self) -> rpc_pb2_grpc.KVStub:
        if self._kv_stub is None:
            channel = await self._get_channel()
            self._kv_stub = rpc_pb2_grpc.KVStub(channel)
        return self._kv_stub

    async def _get_lease(self) -> rpc_pb2_grpc.LeaseStub:
        if self._lease_stub is None:
            channel = await self._get_channel()
            self._lease_stub = rpc_pb2_grpc.LeaseStub(channel)
        return self._lease_stub

    def _should_retry(self, error: grpc.aio.AioRpcError) -> bool:
        return error.code() in self._retryable_statuses

    async def _kv_call(self, fn: Callable[[rpc_pb2_grpc.KVStub], Awaitable[Any]]):
        for attempt in range(2):
            kv = await self._get_kv()
            try:
                return await fn(kv)
            except grpc.aio.AioRpcError as exc:
                if not self._should_retry(exc) or attempt == 1:
                    raise
                await self._reset_channel()
        raise RuntimeError('kv call retries exhausted')

    async def _lease_call(self, fn: Callable[[rpc_pb2_grpc.LeaseStub], Awaitable[Any]]):
        for attempt in range(2):
            lease_stub = await self._get_lease()
            try:
                return await fn(lease_stub)
            except grpc.aio.AioRpcError as exc:
                if not self._should_retry(exc) or attempt == 1:
                    raise
                await self._reset_channel()
        raise RuntimeError('lease call retries exhausted')

    # --- Locking ---

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        owner = client_id or self._default_lock_client
        lease_id = await self._ensure_lock_lease(owner, timeout)
        key = self._lock_key(lock_name)
        cmp = [
            rpc_pb2.Compare(
                key=key,
                target=rpc_pb2.Compare.VERSION,
                result=rpc_pb2.Compare.EQUAL,
                version=0,
            )
        ]
        success = [
            rpc_pb2.RequestOp(
                request_put=rpc_pb2.PutRequest(
                    key=key,
                    value=owner.encode(),
                    lease=lease_id,
                )
            )
        ]
        failure = [
            rpc_pb2.RequestOp(
                request_range=rpc_pb2.RangeRequest(key=key)
            )
        ]
        resp = await self._kv_call(
            lambda stub: stub.Txn(
                rpc_pb2.TxnRequest(compare=cmp, success=success, failure=failure),
                timeout=timeout,
                metadata=self._metadata,
            )
        )
        if resp.succeeded:
            self._lock_keys[(lock_name, owner)] = key
            return True

        if resp.responses and resp.responses[0].response_range.kvs:
            current = resp.responses[0].response_range.kvs[0].value
            if current == owner.encode():
                self._lock_keys[(lock_name, owner)] = key
                return True
        return False

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        owner = client_id or self._default_lock_client
        key = self._lock_keys.pop((lock_name, owner), None) or self._lock_key(lock_name)
        cmp = [
            rpc_pb2.Compare(
                key=key,
                target=rpc_pb2.Compare.VALUE,
                result=rpc_pb2.Compare.EQUAL,
                value=owner.encode(),
            )
        ]
        success = [rpc_pb2.RequestOp(request_delete_range=rpc_pb2.DeleteRangeRequest(key=key))]
        await self._kv_call(
            lambda stub: stub.Txn(rpc_pb2.TxnRequest(compare=cmp, success=success), metadata=self._metadata)
        )

    async def _ensure_lock_lease(self, owner: str, ttl: int) -> int:
        lease = self._session_leases.get(owner)
        if lease is not None:
            await self._keepalive(lease)
            return lease

        lease = self._lock_leases.get(owner)
        if lease is None:
            lease = await self._grant_lease(max(ttl, 5))
            self._lock_leases[owner] = lease
        else:
            await self._keepalive(lease)
        return lease

    # --- Containers ---

    async def allocate_container(self, container_id: str, session_id: str):
        lease_id = self._session_leases.get(session_id)
        if lease_id is None:
            raise ValueError(f'unknown session {session_id}')
        key = self._container_allocation_key(container_id, session_id)
        await self._kv_call(
            lambda stub: stub.Put(
                rpc_pb2.PutRequest(key=key, value=b'1', lease=lease_id),
                metadata=self._metadata,
            )
        )
        await self._increment_uses(container_id)

    async def renew_container(self, container_id: str, session_id: str):
        # refreshed together with session lease
        pass

    async def container_is_allocated(self, container_id: str) -> bool:
        return await self.container_current_uses(container_id) > 0

    async def container_current_uses(self, container_id: str) -> int:
        key = self._container_allocation_prefix(container_id)
        resp = await self._kv_call(
            lambda stub: stub.Range(
                rpc_pb2.RangeRequest(key=key, range_end=self._range_end(key)),
                metadata=self._metadata,
            )
        )
        return resp.count

    async def container_total_uses(self, container_id: str) -> int:
        key = self._container_uses_key(container_id)
        resp = await self._kv_call(
            lambda stub: stub.Range(rpc_pb2.RangeRequest(key=key), metadata=self._metadata)
        )
        if resp.count == 0:
            return 0
        return int(resp.kvs[0].value or b'0')

    async def containers_total_uses_gte(self, min_uses: int) -> List[str]:
        prefix = self._uses_prefix()
        resp = await self._kv_call(
            lambda stub: stub.Range(
                rpc_pb2.RangeRequest(key=prefix, range_end=self._range_end(prefix)),
                metadata=self._metadata,
            )
        )
        matches: List[str] = []
        for kv_pair in resp.kvs:
            uses = int(kv_pair.value or b'0')
            if uses >= min_uses:
                matches.append(kv_pair.key.decode().split(':')[-1])
        return matches

    async def release_container(self, container_id: str, session_id: str):
        await self._kv_call(
            lambda stub: stub.DeleteRange(
                rpc_pb2.DeleteRangeRequest(key=self._container_allocation_key(container_id, session_id)),
                metadata=self._metadata,
            )
        )

    async def remove_container(self, container_id: str):
        prefix = self._container_key_prefix(container_id)
        await self._kv_call(
            lambda stub: stub.DeleteRange(
                rpc_pb2.DeleteRangeRequest(key=prefix, range_end=self._range_end(prefix)),
                metadata=self._metadata,
            )
        )
        await self._kv_call(
            lambda stub: stub.DeleteRange(
                rpc_pb2.DeleteRangeRequest(key=self._container_uses_key(container_id)),
                metadata=self._metadata,
            )
        )

    # --- Sessions ---

    async def generate_session_id(self) -> str:
        lease = await self._grant_lease(self.session_expiry)
        session_id = str(uuid.uuid4())
        self._session_leases[session_id] = lease
        self._lock_leases[session_id] = lease
        return session_id

    async def store_session(self, session_id: str, data: Any):
        lease = self._session_leases.get(session_id)
        if lease is None:
            raise ValueError(f'unknown session {session_id}')
        await self._kv_call(
            lambda stub: stub.Put(
                rpc_pb2.PutRequest(
                    key=self._session_key(session_id),
                    value=json.dumps(data).encode(),
                    lease=lease,
                ),
                metadata=self._metadata,
            )
        )

    async def get_session(self, session_id: str) -> Optional[Any]:
        resp = await self._kv_call(
            lambda stub: stub.Range(
                rpc_pb2.RangeRequest(key=self._session_key(session_id)),
                metadata=self._metadata,
            )
        )
        if resp.count == 0:
            return None
        try:
            return json.loads(resp.kvs[0].value.decode())
        except json.JSONDecodeError:
            return None

    async def renew_session(self, session_id: str):
        lease = self._session_leases.get(session_id)
        if lease is not None:
            await self._keepalive(lease)

    async def delete_session(self, session_id: str):
        await self._kv_call(
            lambda stub: stub.DeleteRange(
                rpc_pb2.DeleteRangeRequest(key=self._session_key(session_id)),
                metadata=self._metadata,
            )
        )
        lease = self._session_leases.pop(session_id, None)
        if lease is not None:
            await self._lease_call(
                lambda stub: stub.LeaseRevoke(
                    rpc_pb2.LeaseRevokeRequest(ID=lease), metadata=self._metadata
                )
            )
        self._lock_leases.pop(session_id, None)

    # --- Helpers ---

    async def _increment_uses(self, container_id: str):
        key = self._container_uses_key(container_id)
        for _ in range(16):
            resp = await self._kv_call(
                lambda stub: stub.Range(rpc_pb2.RangeRequest(key=key), metadata=self._metadata)
            )
            if resp.count == 0:
                compare = [
                    rpc_pb2.Compare(
                        key=key,
                        target=rpc_pb2.Compare.VERSION,
                        result=rpc_pb2.Compare.EQUAL,
                        version=0,
                    )
                ]
                new_value = b'1'
            else:
                kv_pair = resp.kvs[0]
                current = int(kv_pair.value or b'0')
                compare = [
                    rpc_pb2.Compare(
                        key=key,
                        target=rpc_pb2.Compare.MOD,
                        result=rpc_pb2.Compare.EQUAL,
                        mod_revision=kv_pair.mod_revision,
                    )
                ]
                new_value = str(current + 1).encode()
            success = [rpc_pb2.RequestOp(request_put=rpc_pb2.PutRequest(key=key, value=new_value))]
            txn_resp = await self._kv_call(
                lambda stub: stub.Txn(
                    rpc_pb2.TxnRequest(compare=compare, success=success),
                    metadata=self._metadata,
                )
            )
            if txn_resp.succeeded:
                return
        self.logger.warning('failed to increment total uses for container %s', container_id)

    async def _grant_lease(self, ttl: int) -> int:
        resp = await self._lease_call(
            lambda stub: stub.LeaseGrant(rpc_pb2.LeaseGrantRequest(TTL=ttl), metadata=self._metadata)
        )
        return resp.ID

    async def _get_keepalive_stream(self) -> grpc.aio.StreamStreamCall:
        if self._keepalive_stream is None:
            lease_stub = await self._get_lease()
            self._keepalive_stream = lease_stub.LeaseKeepAlive(metadata=self._metadata)
        return self._keepalive_stream

    async def _reset_keepalive_stream(self):
        if self._keepalive_stream is not None:
            try:
                await self._keepalive_stream.done_writing()
            except Exception:
                pass
            self._keepalive_stream = None

    async def _keepalive(self, lease_id: int):
        for _ in range(3):
            stream = await self._get_keepalive_stream()
            try:
                async with self._keepalive_lock:
                    await stream.write(rpc_pb2.LeaseKeepAliveRequest(ID=lease_id))
                    response = await stream.read()
                if response is None or response.TTL == 0:
                    raise RuntimeError('lease expired')
                return
            except Exception:
                await self._reset_keepalive_stream()
                await asyncio.sleep(0.1)
        self.logger.warning('failed to renew lease %s', lease_id)

    def _lock_key(self, name: str) -> bytes:
        return f'{self.prefix}lock:{name}'.encode()

    def _container_key_prefix(self, container_id: str) -> bytes:
        return f'{self.prefix}container:{container_id}'.encode()

    def _container_allocation_prefix(self, container_id: str) -> bytes:
        return self._container_key_prefix(container_id) + b':allocation:'

    def _container_allocation_key(self, container_id: str, session_id: str) -> bytes:
        return self._container_allocation_prefix(container_id) + session_id.encode()

    def _container_uses_key(self, container_id: str) -> bytes:
        return self._uses_prefix() + container_id.encode()

    def _uses_prefix(self) -> bytes:
        return f'{self.prefix}container-uses:'.encode()

    def _session_key(self, session_id: str) -> bytes:
        return f'{self.prefix}session:{session_id}'.encode()

    @staticmethod
    def _range_end(prefix: bytes) -> bytes:
        if not prefix:
            return b'\x00'
        buf = bytearray(prefix)
        for i in range(len(buf) - 1, -1, -1):
            if buf[i] != 0xFF:
                buf[i] += 1
                return bytes(buf[: i + 1])
        return b'\x00'
