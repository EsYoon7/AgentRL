import argparse
import logging

from .config import ConfigLoader
from .task_worker import TaskWorker
from .typings import InstanceFactory

parser = argparse.ArgumentParser(
    prog='python -m agentrl.worker',
    description='The AgentRL task worker',
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument("name", type=str,
                    help="Task name, should match a top-level entry in the configuration file")
parser.add_argument("--config", "-c", type=str, required=True,
                    help="Path to the configuration file")
parser.add_argument("--controller", "-C", type=str, default="http://localhost:5020/api",
                    help="URL of the controller API, e.g., http://localhost:5020/api\n"
                         "Specify in the form of grpc://host:port to enable gRPC transport")
parser.add_argument("--self", "-s", type=str, default="http://localhost:5021/api",
                    help="URL of this task worker's API, e.g., http://localhost:5021/api\n"
                         "Must be reachable by the controller unless gRPC transport is enabled")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", "-p", type=int, default=5021)
parser.add_argument("--log-level", type=str, default="INFO")

args = parser.parse_args()

logging.basicConfig(
    level=args.log_level.upper(),
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %Z',
)

logger = logging.getLogger('task_worker')
logger.info(f'Starting task {args.name} {args.config=} {args.controller=} {args.self=} {args.port=}')

conf = ConfigLoader().load_from(args.config, args.name)
logger.info(f'Resolved configuration: {conf[args.name]}')
asyncio_task = InstanceFactory.model_validate(conf[args.name]).create()

task_worker = TaskWorker(
    asyncio_task,
    controller_address=args.controller,
    self_address=args.self,
    logger=logger
)
task_worker.run(host=args.host, port=args.port)
