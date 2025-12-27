import json
import os
from copy import deepcopy
from typing import Set, Dict, Any, Optional

import yaml

from .consul import ConsulConfigLoader


class ConfigLoader:

    def __init__(self) -> None:
        self.loading: Set[str] = set()
        self.loaded: Dict[str, Any] = dict()

        self._consul_loader = ConsulConfigLoader()

    def load_from(self, path, name: Optional[str] = None) -> Dict:
        # 0. base config
        config = self._load_file(path)

        # 1. file override
        override_path = os.environ.get("AGENTRL_CONFIG_OVERRIDE")
        if override_path:
            override_config = self._load_file(override_path)
            config = self.deep_merge(config, override_config)

        # 2. consul override
        if name is not None:
            consul_override = self._consul_loader.get_config(name)
            if consul_override:
                v = config.get(name) or {}
                config[name] = self.deep_merge(v, consul_override)

        # 3. environment variable override
        config = self.deep_override_from_env(config)

        return config

    def _load_file(self, path):
        path = os.path.realpath(path)
        if path in self.loading:
            raise Exception("Circular import detected: {}".format(path))
        if path in self.loaded:
            return deepcopy(self.loaded[path])
        if not os.path.exists(path):
            raise Exception("File not found: {}".format(path))
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path) as f:
                config = yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path) as f:
                config = json.load(f)
        else:
            raise Exception("Unknown file type: {}".format(path))
        self.loading.add(path)
        try:
            config = self.parse_imports(os.path.dirname(path), config)
        except Exception as e:
            self.loading.remove(path)
            raise e
        config = self.parse_default_and_overwrite(deepcopy(config))
        self.loading.remove(path)
        self.loaded[path] = config
        return self.loaded[path]

    def parse_imports(self, path, raw_config):
        raw_config = deepcopy(raw_config)
        if isinstance(raw_config, dict):
            ret = {}
            if "import" in raw_config:
                v = raw_config.pop("import")
                if isinstance(v, str):
                    config = self.load_from(os.path.join(path, v))
                    ret = self.deep_merge(ret, config)
                elif isinstance(v, list):
                    for vv in v:
                        assert isinstance(
                            vv, str
                        ), "Import list must be a list of strings, found {}".format(
                            type(vv)
                        )
                        config = self.load_from(os.path.join(path, vv))
                        ret = self.deep_merge(ret, config)
                else:
                    raise Exception("Unknown import value: {}".format(v))
            for k, v in raw_config.items():
                raw_config[k] = self.parse_imports(path, v)
            ret = self.deep_merge(ret, raw_config)
            return ret
        elif isinstance(raw_config, list):
            ret = []
            for v in raw_config:
                ret.append(self.parse_imports(path, v))
            return ret
        else:
            return raw_config

    def parse_default_and_overwrite(self, config):
        if isinstance(config, dict):
            if not config:
                return {}
            ret = {}
            overwriting = False
            defaulting = False
            if "overwrite" in config:
                overwrite = self.parse_default_and_overwrite(config.pop("overwrite"))
                overwriting = True
            if "default" in config:
                default = self.parse_default_and_overwrite(config.pop("default"))
                defaulting = True
            for k, v in config.items():
                parsed_v = self.parse_default_and_overwrite(v)
                if overwriting:
                    parsed_v = self.deep_merge(parsed_v, overwrite)
                if defaulting:
                    parsed_v = self.deep_merge(default, parsed_v)
                ret[k] = parsed_v
            return ret
        return config

    def deep_override_from_env(self, config, prefix = ''):
        if isinstance(config, dict):
            ret = {}
            for k, v in config.items():
                env_key = f'{prefix}{k}'.replace('-', '_').upper()
                if env_key in os.environ:
                    try:
                        if isinstance(v, bool):
                            ret[k] = os.environ[env_key].lower() in ['true', '1', 'yes']
                        else:
                            ret[k] = type(v)(os.environ[env_key])
                    except Exception:
                        ret[k] = os.environ[env_key]
                else:
                    ret[k] = self.deep_override_from_env(v, f'{env_key}_')
            return ret
        return config

    def deep_merge(self, base_item, new_item):
        if isinstance(base_item, dict) and isinstance(new_item, dict):
            ret = deepcopy(base_item)
            for key in new_item:
                if key in ret:
                    ret[key] = self.deep_merge(ret[key], new_item[key])
                else:
                    ret[key] = new_item[key]
            return ret
        return new_item
