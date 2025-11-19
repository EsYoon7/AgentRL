# Workaround for https://github.com/protocolbuffers/protobuf/issues/1491

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import common_pb2
import common_pb2_grpc
import controller_v1.controller_pb2 as controller_v1_pb2
import controller_v1.controller_pb2_grpc as controller_v1_pb2_grpc
import snapshots_v1.snapshots_pb2 as snapshots_v1_pb2
import snapshots_v1.snapshots_pb2_grpc as snapshots_v1_pb2_grpc
