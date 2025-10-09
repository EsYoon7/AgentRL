# Workaround for https://github.com/protocolbuffers/protobuf/issues/1491

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import common_pb2
import common_pb2_grpc
import controller_pb2
import controller_pb2_grpc
