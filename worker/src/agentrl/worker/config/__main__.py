import argparse
import json

import yaml

from .loader import ConfigLoader

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="Config file to load")
# output format: choice from json or yaml
parser.add_argument(
    "--output", "-o", choices=["json", "yaml"], default="yaml", help="Output format"
)
args = parser.parse_args()
config = ConfigLoader().load_from(args.config)
if args.output == "json":
    print(json.dumps(config, indent=2))
elif args.output == "yaml":
    print(yaml.dump(config))
