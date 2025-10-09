from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ._base import EnvironmentController


class EnvironmentDelegation:
    """
    Task-specific environment configuration.
    """

    controller: EnvironmentController

    def __init__(self, name: str):
        self.name = name  # name to identify the task, might be added in labels etc.

    def get_name(self) -> str:
        return self.name

    def get_subtypes(self) -> List[str]:
        return ['default']

    def get_service_port(self, subtype: str) -> Optional[Union[int, List[int]]]:
        """
        If the task environments exposes a service on a port,
        implement this method to return the port number.
        """
        return None

    def is_exclusive(self, subtype: str) -> bool:
        """
        Checks if the task requires the environment to be exclusively used by only one session
        If returns True, the environment is strictly bond to individual session.
        Deprecated: use `get_concurrency_limit` and `get_reuse_limit` instead.
        """
        return False

    def supports_reuse(self, subtype: str) -> bool:
        """
        Checks if the task supports reusing the environment of this subtype.
        If returns True, the environments of this subtype will not be recreated no matter
        the task specifies it is immutable or not.
        Deprecated: use `get_concurrency_limit` and `get_reuse_limit` instead.
        """
        return False

    async def create_docker_container(self, attrs: dict, subtype: str) -> dict:
        """
        If the task supports running in a container through the Docker API,
        implement this method to manipulate the container attributes.

        Task should call the controller to create more containers if there are dependencies.

        :param attrs: The attributes of the container to create. For easier manipulation,
                      the `Env` are stored as a dict instead of a list as defined in the Docker API.
        :param subtype: The subtype of the environment to create.
        """
        raise NotImplementedError

    async def post_create_docker_container(self, subtype: str, environment_id: str, environment_url: str):
        """
        Hook to perform additional actions after the Docker container is created.
        Only called if the final delegation class directly overrides this method.
        If health check is configured, only called if the container is healthy.
        """
        raise NotImplementedError

    def get_container_images(self) -> Dict[str, str]:
        """
        If the task supports running in a container,
        implement this method to return the mapping of subtypes to image names or IDs.
        """
        raise NotImplementedError

    def has_homepage(self) -> bool:
        """
        Check if the task has a homepage.
        If there is, a homepage is created after all other subtypes of environments are allocated to the session,
        with the environment variables provided by `get_homepage_envs`.
        """
        return False

    def get_homepage_subtype(self) -> str:
        """
        Get the name of the homepage subtype.
        This method is called only if `has_homepage` returns True.
        """
        raise NotImplementedError

    def get_homepage_envs(self, site_urls: Dict[str, str]) -> dict:
        """
        Get the environment variables to set for the homepage from the URLs of each site allocated.
        This method is called only if `has_homepage` returns True.
        """
        raise NotImplementedError

    def get_concurrency_limit(self, subtype: str) -> int:
        """
        Get the maximum number of concurrent sessions that can use a single container
        Use 0 for unlimited concurrency. Defaults to use `is_exclusive` for compatibility.
        """
        return 1 if self.is_exclusive(subtype) else 0

    def get_reuse_limit(self, subtype: str) -> int:
        """
        Get the maximum number of times a container can be reused before being recreated
        Use 0 for unlimited reuse. Defaults to use `is_exclusive` and `supports_reuse` for compatibility.
        """
        return 1 if self.is_exclusive(subtype) or not self.supports_reuse(subtype) else 0
