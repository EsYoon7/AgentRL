import base64
import json
import logging
import os
from typing import Optional

import httpx
import yaml


class ConsulConfigLoader:

    def __init__(self,
                 base_url: str = os.getenv('CONSUL_HTTP_ADDR'),
                 token: Optional[str] = os.getenv('CONSUL_HTTP_TOKEN')):
        self.logger = logging.getLogger(__name__)

        self.base_url = base_url
        self.token = token

    def get_config(self, name: str) -> Optional:
        if not self.base_url:
            return None
        try:
            response = httpx.get(f'{self.base_url}/v1/kv/agentrl/config/{name}', headers={
                'X-Consul-Token': self.token
            } if self.token else {})
            response.raise_for_status()
            raw = base64.b64decode(response.json()[0]['Value']).decode('utf-8')
            try:
                return json.loads(raw)
            except ValueError:
                return yaml.safe_load(raw)
        except Exception:
            self.logger.exception(f'Consul base url is configured, but failed to fetch override config for {name}')
            return None
