from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, List, Optional, Union

import httpx

from ._base import StateProvider
from ...utils import parse_duration


class ConsulStateProvider(StateProvider):
    """
    This is an experimental state provider that is not battle-tested. Use with caution.
    """

    def __init__(self,
                 connection: str,
                 token: Optional[str],
                 datacenter: Optional[str],
                 namespace: Optional[str],
                 prefix: str):
        self.logger = logging.getLogger(__name__)

        self._client: Optional[httpx.AsyncClient] = None
        self._client_session: Optional[str] = None
        self._client_session_expires_at: Optional[float] = None
        self._client_base_url = connection.rstrip('/')
        self._client_token = token

        self.prefix = (prefix + '/') if prefix else ''
        self.datacenter = datacenter
        self.namespace = namespace

        self.session_expiry = 600

    # --- Client ---

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = httpx.AsyncHTTPTransport(retries=3)
            timeout = httpx.Timeout(10.0, connect=5.0)
            self._client = httpx.AsyncClient(
                base_url=self._client_base_url,
                headers={'X-Consul-Token': self._client_token} if self._client_token else None,
                params={'ns': self.namespace, 'dc': self.datacenter},
                transport=transport,
                timeout=timeout,
            )
        return self._client

    async def _reset_client(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        client = await self._get_client()
        try:
            return await client.request(method, url, **kwargs)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            self.logger.warning('consul %s %s failed (%s), retrying once', method.upper(), url, exc)
            await self._reset_client()
            client = await self._get_client()
            return await client.request(method, url, **kwargs)

    async def _get_client_session(self, timeout: int) -> str:
        if self._client_session is None or self._client_session_expires_at is None:
            remaining_time = 0
        else:
            remaining_time = self._client_session_expires_at - time.time()

        if remaining_time <= 0:
            self._client_session = await self._session_create({
                'LockDelay': '2s',
                'Name': f'{self.prefix}client',
                'Behavior': 'delete',
                'TTL': f'{max(10, timeout)}s',  # Consul TTL must be >= 10s
            })
            self._client_session_expires_at = time.time() + max(10, timeout)
            self.logger.info('created client session %s', self._client_session)
        elif remaining_time < timeout - 2:
            expiry = await self._session_renew(self._client_session)
            if expiry is None:
                self._client_session = None
                self._client_session_expires_at = None
                return await self._get_client_session(timeout)
            self._client_session_expires_at = time.time() + expiry
            self.logger.info('renewed client session %s', self._client_session)

        return self._client_session

    async def _kv_get(self, key: str, **kwargs) -> Optional[Union[dict, List[str], List[dict]]]:
        response = await self._request('GET', f'/v1/kv/{key}', params=kwargs)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            if len(data) == 1 and isinstance(data[0], dict):
                return data[0]
            else:
                return data
        return None

    async def _kv_put(self, key: str, content: Union[str, bytes], **kwargs) -> bool:
        response = await self._request('PUT', f'/v1/kv/{key}', content=content, params=kwargs)
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()
        return response.json() is True

    async def _kv_delete(self, key: str, **kwargs):
        response = await self._request('DELETE', f'/v1/kv/{key}', params=kwargs)
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()

    async def _session_create(self, payload: dict) -> str:
        response = await self._request('PUT', '/v1/session/create', json=payload)
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()
        data = response.json()
        return data['ID']

    async def _session_renew(self, session_id: str) -> Optional[float]:
        response = await self._request('PUT', f'/v1/session/renew/{session_id}')
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return parse_duration(data[0]['TTL'], return_seconds=True)
        return None

    async def _session_delete(self, session_id: str):
        response = await self._request('PUT', f'/v1/session/destroy/{session_id}')
        if response.status_code >= 400:
            self.logger.error('consul returned error: %s', response.text)
        response.raise_for_status()

    # --- Locking ---

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        key = self._lock_key(lock_name)
        if client_id is None:
            session_id = await self._get_client_session(timeout)
        else:
            session_id = client_id
        return await self._kv_put(key, '1', acquire=session_id)

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        key = self._lock_key(lock_name)
        if client_id is None:
            session_id = self._client_session
        else:
            session_id = client_id
        if session_id is None:
            return
        await self._kv_put(key, '0', release=session_id)

    # --- Container Allocation ---

    async def allocate_container(self, container_id: str, session_id: str):
        alloc_key = self._container_allocation_key(container_id, session_id)
        ok = await self._kv_put(alloc_key, '1', acquire=session_id)
        if not ok:
            raise RuntimeError(f'failed to allocate container {container_id} for session {session_id}')

        # increment total uses via CAS loop
        uses_key = self._container_uses_key(container_id)
        for _ in range(16):
            meta = await self._kv_get(uses_key)
            if not meta:
                ok = await self._kv_put(uses_key, '1', cas=0)
                if ok:
                    break
                await asyncio.sleep(0)
                continue
            current = int(base64.b64decode(meta['Value'] or b'0').decode())
            ok = await self._kv_put(uses_key, str(current + 1), cas=meta['ModifyIndex'])
            if ok:
                break
            await asyncio.sleep(0)
        else:
            self.logger.warning('failed to increment total uses for container %s', container_id)

    async def renew_container(self, container_id: str, session_id: str):
        pass  # covered by session renewals

    async def container_is_allocated(self, container_id: str) -> bool:
        return await self.container_current_uses(container_id) > 0

    async def container_current_uses(self, container_id: str) -> int:
        prefix = self._container_allocation_key_prefix(container_id)
        keys = await self._kv_get(prefix, keys=True)
        return len(keys or [])

    async def container_total_uses(self, container_id: str) -> int:
        meta = await self._kv_get(self._container_uses_key(container_id))
        if not meta:
            return 0
        return int(base64.b64decode(meta['Value'] or b'0').decode())

    async def containers_total_uses_gte(self, threshold: int) -> List[str]:
        result: List[str] = []
        metas = await self._kv_get(f'{self.prefix}container-uses/', recurse=True) or []
        for meta in metas:
            container_id = meta['Key'].split('/')[-1]
            value = int(base64.b64decode(meta['Value'] or b'0').decode())
            if value >= threshold:
                result.append(container_id)
        return result

    async def release_container(self, container_id: str, session_id: str):
        key = self._container_allocation_key(container_id, session_id)
        released = await self._kv_put(key, '0', release=session_id)
        if released:
            await self._kv_delete(key)

    async def remove_container(self, container_id: str):
        await self._kv_delete(self._container_key_prefix(container_id), recurse=True)
        await self._kv_delete(self._container_uses_key(container_id))

    # --- Session Data ---

    async def generate_session_id(self) -> str:
        return await self._session_create({
            'LockDelay': '0s',
            'Name': f'{self.prefix}session',
            'Behavior': 'delete',
            'TTL': f'{self.session_expiry}s'
        })

    async def store_session(self, session_id: str, data: Any):
        key = self._session_key(session_id)
        await self._kv_put(key, json.dumps(data), acquire=session_id)

    async def get_session(self, session_id: str) -> Optional[Any]:
        meta = await self._kv_get(self._session_key(session_id))
        if not meta:
            return None
        try:
            return json.loads(base64.b64decode(meta['Value']).decode())
        except ValueError:
            return None

    async def renew_session(self, session_id: str):
        await self._session_renew(session_id)

    async def delete_session(self, session_id: str):
        await self._session_delete(session_id)

    async def close(self):
        await self._reset_client()

    # --- Key Names ---

    def _lock_key(self, lock_name: str) -> str:
        return f'{self.prefix}lock/{lock_name}'

    def _container_key_prefix(self, container_id: str) -> str:
        return f'{self.prefix}container/{container_id}/'

    def _container_allocation_key_prefix(self, container_id: str) -> str:
        return f'{self._container_key_prefix(container_id)}allocation/'

    def _container_allocation_key(self, container_id: str, session_id: str) -> str:
        return f'{self._container_allocation_key_prefix(container_id)}{session_id}'

    def _container_uses_key(self, container_id: str) -> str:
        return f'{self.prefix}container-uses/{container_id}'

    def _session_key(self, session_id: str) -> str:
        return f'{self.prefix}session/{session_id}'
