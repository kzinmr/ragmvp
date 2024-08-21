"""Base schema for data structures."""

import json
import logging
import pickle
import textwrap
import uuid
from enum import Enum
from typing import Any, Self

from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"
# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70


logger = logging.getLogger(__name__)

SAMPLE_TEXT = """
Context
LLMs are a phenomenal piece of technology for knowledge generation and reasoning.
They are pre-trained on large amounts of publicly available data.
How do we best augment LLMs with our own private data?
We need a comprehensive toolkit to help perform this data augmentation for LLMs.

Proposed Solution
That's where LlamaIndex comes in. LlamaIndex is a "data framework" to help
you build LLM  apps. It provides the following tools:

Offers data connectors to ingest your existing data sources and data formats
(APIs, PDFs, docs, SQL, etc.)
Provides ways to structure your data (indices, graphs) so that this data can be
easily used with LLMs.
Provides an advanced retrieval/query interface over your data:
Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
Allows easy integrations with your outer application framework
(e.g. with LangChain, Flask, Docker, ChatGPT, anything else).
LlamaIndex provides tools for both beginner users and advanced users.
Our high-level API allows beginner users to use LlamaIndex to ingest and
query their data in 5 lines of code. Our lower-level APIs allow advanced users to
customize and extend any module (data connectors, indices, retrievers, query engines,
reranking modules), to fit their needs.
"""


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class BaseComponent(BaseModel):
    """Base component object to capture class names."""

    class_name: str = Field(default_factory=lambda: "base_component")
    model_config = ConfigDict(extra="allow")

    @classmethod
    def __json_schema_extra__(
        cls, schema: dict[str, Any], model: "BaseComponent"
    ) -> None:
        """Add class name to schema."""
        schema["properties"]["class_name"] = {
            "title": "Class Name",
            "type": "string",
            "default": model.class_name,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.dict()

        # remove attributes that are not pickleable -- kind of dangerous
        keys_to_remove = []
        for key, val in state.items():
            try:
                pickle.dumps(val)
            except Exception:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            logging.warning("Removing unpickleable attribute {%s}", key)
            del state[key]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Use the __dict__ and __init__ method to set state
        # so that all variables initialize
        try:
            self.__init__(**state)
        except Exception:
            # Fall back to the default __setstate__ method
            # This may not work if the class had unpickleable attributes
            object.__setstate__(self, "__dict__", state)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"


class Document(BaseModel):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    model_config = ConfigDict(populate_by_name=True)

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
    )

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering
    """
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )
    excluded_embed_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )

    text: str = Field(default="", description="Text content of the node.")
    mimetype: str = Field(
        default="text/plain", description="MIME type of the node content."
    )
    start_char_idx: int | None = Field(
        default=None, description="Start char index of the node."
    )
    end_char_idx: int | None = Field(
        default=None, description="End char index of the node."
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        return self.text_template.format(
            content=self.text, metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.text = value

    def get_node_info(self) -> dict[str, Any]:
        """Get node info."""
        return {"start": self.start_char_idx, "end": self.end_char_idx}

    def get_text(self) -> str:
        return self.get_content(metadata_mode=MetadataMode.NONE)

    def __str__(self) -> str:
        source_text_truncated = truncate_text(
            self.get_content().strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Doc ID: {self.id_}\n{source_text_wrapped}"

    def to_langchain_format(self) -> "LCDocument":
        """Convert struct to LangChain document format."""
        metadata = self.metadata or {}
        return LCDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_format(cls, doc: "LCDocument") -> "Document":
        """Convert struct from LangChain document format."""
        return cls(text=doc.page_content, metadata=doc.metadata)

    @classmethod
    def example(cls) -> "Document":
        return Document(
            text=SAMPLE_TEXT,
            metadata={"filename": "README.md", "category": "codebase"},
        )
