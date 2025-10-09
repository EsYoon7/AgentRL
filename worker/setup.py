import os
import shutil
from pathlib import Path

from grpc_tools import protoc
from setuptools import setup, Command
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.sdist import sdist as _sdist


class _GenerateProtoFiles(Command):

    description = 'Generate Python gRPC stubs from .proto'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        well_known_protos_path = Path(protoc.__file__).resolve().parent / '_proto'

        root_dir = Path(__file__).resolve().parent
        # Prefer local ./proto (present inside sdist) else fall back to ../proto (repo)
        local_proto = root_dir / 'proto'
        repo_proto = root_dir.parent / 'proto'
        source_dir = local_proto if local_proto.exists() else repo_proto

        if not source_dir.exists():
            raise RuntimeError(f'Could not locate proto directory. Tried: {local_proto} and {repo_proto}')

        output_dir = root_dir / 'src' / 'agentrl' / 'worker' / 'pb'
        os.makedirs(output_dir, exist_ok=True)

        protos = [str(p) for p in source_dir.rglob('**/*.proto')]
        if not protos:
            print(f'No .proto files found under {source_dir}; skipping codegen.')
            return

        proto_args = [
            'grpc_tools.protoc',
            f'--proto_path={well_known_protos_path}',
            f'--proto_path={source_dir}',
            f'--python_out={output_dir}',
            f'--grpc_python_out={output_dir}',
            f'--pyi_out={output_dir}',
            *protos,
        ]
        print(f'running {" ".join(proto_args)}')
        if protoc.main(proto_args) != 0:
            raise RuntimeError('failed to compile protos')


class _BuildPy(build_py):

    def run(self):
        self.run_command('generate_protos')
        super().run()


class _Develop(develop):

    def run(self):
        self.run_command('generate_protos')
        super().run()


class _Sdist(_sdist):

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        base = Path(base_dir)
        pkg_dir = Path(__file__).resolve().parent  # .../worker
        repo_proto = pkg_dir.parent / 'proto'
        dest = base / 'proto'

        if not repo_proto.exists():
            raise RuntimeError(f'Expected repo proto directory missing: {repo_proto}')

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(repo_proto, dest)


setup(
    cmdclass={
        'build_py': _BuildPy,
        'develop': _Develop,
        'sdist': _Sdist,
        'generate_protos': _GenerateProtoFiles,
    }
)
