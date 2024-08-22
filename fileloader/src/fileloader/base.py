"""Base reader class."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fileloader.langchain import LCDocument

from fileloader.schema import Document


class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        msg = f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        raise NotImplementedError(msg)

    async def alazy_load_data(
        self, *args: Any, **load_kwargs: Any
    ) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        # Fake async - just calls the sync method. Override in subclasses for real async implementations.
        return self.lazy_load_data(*args, **load_kwargs)

    @abstractmethod
    def load_data(self, *args: Any, **load_kwargs: Any) -> list[Document]:
        """Load data from the input directory."""

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> list[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)

    def load_langchain_documents(self, **load_kwargs: Any) -> list["LCDocument"]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]


class ResourcesReaderMixin(ABC):
    """Mixin for readers that provide access to different types of resources.

    Resources refer to specific data entities that can be accessed by the reader.
    Examples of resources include files for a filesystem reader, channel IDs for a Slack reader, or pages for a Notion reader.
    """

    @abstractmethod
    def list_resources(self, *args: Any, **kwargs: Any) -> list[str]:
        """list of identifiers for the specific type of resources available in the reader.

        Returns:
            list[str]: list of identifiers for the specific type of resources available in the reader.

        """

    async def alist_resources(self, *args: Any, **kwargs: Any) -> list[str]:
        """list of identifiers for the specific type of resources available in the reader asynchronously.

        Returns:
            list[str]: A list of resources based on the reader type, such as files for a filesystem reader,
            channel IDs for a Slack reader, or pages for a Notion reader.

        """
        return self.list_resources(*args, **kwargs)

    @abstractmethod
    def get_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Get a dictionary of information about a specific resource.

        Args:
            resource_id (str): The resource identifier.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary of information about the resource.

        """

    async def aget_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Get a dictionary of information about a specific resource asynchronously.

        Args:
            resource_id (str): The resource identifier.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary of information about the resource.

        """
        return self.get_resource_info(resource_id, *args, **kwargs)

    def list_resources_with_info(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        """Get a dictionary of information about all resources.

        Returns:
            dict[str, dict]: A dictionary of information about all resources.

        """
        return {
            resource: self.get_resource_info(resource, *args, **kwargs)
            for resource in self.list_resources(*args, **kwargs)
        }

    async def alist_resources_with_info(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        """Get a dictionary of information about all resources asynchronously.

        Returns:
            dict[str, dict]: A dictionary of information about all resources.

        """
        return {
            resource: await self.aget_resource_info(resource, *args, **kwargs)
            for resource in await self.alist_resources(*args, **kwargs)
        }

    @abstractmethod
    def load_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> list[Document]:
        """Load data from a specific resource.

        Args:
            resource_id (str): The resource identifier.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[Document]: A list of documents loaded from the resource.

        """

    async def aload_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> list[Document]:
        """Read file from filesystem and return documents asynchronously."""
        return self.load_resource(resource_id, *args, **kwargs)

    def load_resources(
        self, resource_ids: list[str], *args: Any, **kwargs: Any
    ) -> list[Document]:
        """Similar to load_data, but only for specific resources.

        Args:
            resource_ids (list[str]): list of resource identifiers.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[Document]: A list of documents loaded from the resources.

        """
        return [
            doc
            for resource in resource_ids
            for doc in self.load_resource(resource, *args, **kwargs)
        ]

    async def aload_resources(
        self, resource_ids: list[str], *args: Any, **kwargs: Any
    ) -> dict[str, list[Document]]:
        """Similar ato load_data, but only for specific resources.

        Args:
            resource_ids (list[str]): list of resource identifiers.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict[str, list[Document]]: A dictionary of documents loaded from the resources.

        """
        return {
            resource: await self.aload_resource(resource, *args, **kwargs)
            for resource in resource_ids
        }
