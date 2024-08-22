import tempfile
import uuid
import warnings
from collections.abc import Iterable
from datetime import datetime
from functools import cached_property
from typing import Any, Literal

import lancedb
import numpy as np
import openai
from lancedb.embeddings import TextEmbeddingFunction, get_registry
from lancedb.embeddings.registry import register
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import Reranker
from lancedb.table import LanceTable, Table
from pydantic import BaseModel, ConfigDict, Field

# from langchain_core.documents import Document
from searchapi.config import get_settings


@register("azure_openai")
class AzureOpenAIEmbeddings(TextEmbeddingFunction):
    """An embedding function that uses the Azure OpenAI API"""

    name: str = "text-embedding-ada-002"
    azure_api_key: str
    azure_endpoint: str
    azure_deployment: str
    azure_api_version: str
    dim: int | None = None

    @cached_property
    def _ndims(self) -> int:
        match self.name:
            case "text-embedding-ada-002":
                return 1536
            case "text-embedding-3-large":
                return self.dim or 3072
            case "text-embedding-3-small":
                return self.dim or 1536
            case _:
                msg = f"Unknown model name {self.name}"
                raise ValueError(msg)

    def ndims(self) -> int:
        return self._ndims

    def generate_embeddings(self, texts: list[str] | np.ndarray) -> list[np.array]:
        """Get the embeddings for the given texts

        Parameters
        ----------
        texts: list[str] or np.ndarray (of str)
            The texts to embed

        """
        if self.name == "text-embedding-ada-002":
            rs = self._openai_client.embeddings.create(input=texts, model=self.name)
        else:
            rs = self._openai_client.embeddings.create(
                input=texts, model=self.name, dimensions=self.ndims()
            )
        return [v.embedding for v in rs.data]

    @cached_property
    def _openai_client(self) -> openai.AzureOpenAI:
        # from lancedb.util import attempt_import_or_raise
        # openai = attempt_import_or_raise("openai")
        # if not os.environ.get("OPENAI_API_KEY"):
        #     api_key_not_found_help("openai")
        return openai.AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_deployment=self.azure_deployment,
        )


embedding_func: AzureOpenAIEmbeddings = (
    get_registry()
    .get("azure_openai")
    .create(
        name="text-embedding-3-large",
        azure_api_key=get_settings().azure_openai_api_key,
        azure_endpoint=get_settings().azure_openai_endpoint,
        azure_deployment=get_settings().azure_openai_embedding_deployment,
        azure_api_version=get_settings().azure_openai_embedding_api_version,
    )
)


class Document(LanceModel):
    id: str | None
    text: str = embedding_func.SourceField()
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField(default=None)  # type: ignore[reportInvalidTypeForm]
    file_path: str | None
    file_name: str | None
    file_type: str | None
    file_size: int | None
    creation_date: datetime | None
    last_modified_date: datetime | None
    # last_accessed_date: datetime | None  # fsspec does not support atime
    total_pages: int | None
    page_number: int | None

    @classmethod
    def metadata_fields(cls) -> list[str]:
        return [f for f in cls.model_fields if f not in ("id", "text", "vector")]


class RawDocument(BaseModel):
    id: str | None
    text: str
    file_path: str | None
    file_name: str | None
    file_type: str | None
    file_size: int | None
    creation_date: datetime | None  # ctime
    last_modified_date: datetime | None  # mtime
    # last_accessed_date: datetime | None  # fsspec does not support atime
    total_pages: int | None
    page_number: int | None

    model_config = ConfigDict(extra="ignore")


class LCDocumentLike(BaseModel):
    id: str | None = None
    metadata: dict = Field(default_factory=dict)
    page_content: str
    vector: list[float]
    type: Literal["Document"] = "Document"

    model_config = ConfigDict(extra="ignore")


PA_TABLE = Any  # pa.Table: Typing is unavailable due to Cython implementation
LANCEDB_QUERY_TYPE = Literal["vector", "fts", "hybrid", "auto"]
LANCEDB_METRIC_TYPE = Literal["L2", "cosine"]
LANCEDB_RESPONSE = list[tuple[LCDocumentLike, float]]


class LanceDBSearchArguments(BaseModel):
    k: int | None = None
    filter_dict: dict[str, str] | None = None
    name: str | None = None
    prefilter: bool = False
    metrics: LANCEDB_METRIC_TYPE = "L2"
    query_type: LANCEDB_QUERY_TYPE = "vector"

    # model_config = ConfigDict(extra="forbid")


class LanceDB:
    DEFAULT_K = 4

    def __init__(
        self,
        table_name: str | None = None,
        uri: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize with Lance DB vectorstore"""
        self.is_fts_indexed = False
        self._reranker = reranker

        # uri / connection / table

        if uri is None:
            uri = tempfile.mkdtemp()
        self._db: lancedb.db.LanceDBConnection = lancedb.connect(uri)
        # if isinstance(connection, lancedb.db.LanceDBConnection):
        #     self._db = connection
        # else:
        #     self._db: lancedb.db.LanceDBConnection = lancedb.connect(uri)
        _mode: Literal["overwrite", "create"] = "overwrite"
        if table_name is None:
            table_name = get_settings().lancedb_table
        self._table = self._db.create_table(table_name, schema=Document, mode=_mode)
        self._table_name = table_name

    def _get_table(self, name: str | None = None) -> Table:
        """Fetches a table object from the database.

        Args:
            name (str, optional): The name of the table to fetch. Defaults to None
                                and fetches current table object.

        Returns:
            Table: The fetched table object.

        Raises:
            ValueError: If the specified table is not found in the database.

        """
        if name is None:
            return self._table

        try:
            return self._db.open_table(name)
        except Exception:
            msg = f"Table `{name}` not found in the database."
            return ValueError(msg)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Turn texts into embedding and add it to the database
        TOCHECK: if it handle rate limit for large documents

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            list of ids of the added texts.

        """
        _texts = list(texts)
        _ids = ids or [str(uuid.uuid4()) for _ in _texts]
        _metadatas = metadatas if metadatas is not None else [{} for _ in _texts]
        # embeddings = embedding_func.compute_source_embeddings(_texts)
        raw_docs = [
            RawDocument(id=_id, text=_text, **_metadata).model_dump()
            for _id, _text, _metadata in zip(_ids, _texts, _metadatas, strict=True)
        ]

        self._table.add(raw_docs, mode="overwrite")
        self.is_fts_indexed = False

        return ids

    def create_scalar_index(
        self,
        col_name: str,
        table_name: str | None = None,
    ) -> None:
        """Create a scalar(for non-vector cols) or a vector index on a table.
        Make sure your vector column has enough data before creating an index on it.

        Args:
            col_name: Provide if you want to create index on a non-vector column.
            table_name: Name of the table to create index on. Defaults to None.

        Returns:
            None

        """
        tbl = self._get_table(table_name)
        if tbl:
            tbl.create_scalar_index(col_name)

    def create_vector_index(
        self,
        num_partitions: int | None = 256,
        num_sub_vectors: int | None = 96,
        index_cache_size: int | None = None,
        metric: LANCEDB_METRIC_TYPE | None = "L2",
        table_name: str | None = None,
    ) -> None:
        """Create a scalar(for non-vector cols) or a vector index on a table.
        Make sure your vector column has enough data before creating an index on it.

        Args:
            col_name: Provide if you want to create index on a non-vector column.
            metric: Provide the metric to use for vector index. Defaults to 'L2'
                    choice of metrics: 'L2' or 'cosine'
            num_partitions: Number of partitions to use for the index. Defaults to 256.
            num_sub_vectors: Number of sub-vectors to use for the index. Defaults to 96.
            index_cache_size: Size of the index cache. Defaults to None.
            table_name: Name of the table to create index on. Defaults to None.

        Returns:
            None

        """
        tbl = self._get_table(table_name)
        if tbl:
            tbl.create_index(
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_cache_size=index_cache_size,
            )

    def create_fts_index(
        self,
        field_names: str | list[str] | None = None,
        table_name: str | None = None,
    ) -> None:
        tbl = self._get_table(table_name)
        if tbl is None:
            msg = "Table not found in the database."
            raise ValueError(msg)

        assert isinstance(tbl, LanceTable)
        if field_names is None:
            field_names = "text"
        tbl.create_fts_index(field_names, replace=True)
        self.is_fts_indexed = True

    def _response_postprocess(self, results: PA_TABLE) -> LANCEDB_RESPONSE:
        """Convert PyArrow results to documents."""
        columns = results.schema.names  # pa.Schema

        if "_distance" in columns:
            score_col = "_distance"
        elif "_relevance_score" in columns:
            score_col = "_relevance_score"
        else:
            msg = "No score column found in the results."
            raise ValueError(msg)

        return [
            (
                LCDocumentLike(
                    page_content=results["text"][idx].as_py(),
                    metadata={
                        key: results[key][idx].as_py()
                        for key in Document.metadata_fields()
                    },
                    vector=results["vector"].to_pylist()[idx],
                ),
                results[score_col][idx].as_py(),
            )
            for idx in range(len(results))
        ]

    def _query(
        self,
        query: list[float] | str | tuple[list[float], str],
        args: LanceDBSearchArguments,
    ) -> PA_TABLE:
        """LanceDB query function.

        Args:
            query (list[float] | tuple[list[float], str]): Query to search for.
            args (LanceDBSearchArguments): Search arguments:
                k (int | None, optional): Number of documents to return. Defaults to 4.
                filter_dict (dict[str, str] | None, optional): Filter by metadata.
                name (str | None, optional): Name of the table to search in.
                prefilter (bool, optional): Whether to apply the filter prior to the vector search.
                metrics (str, optional): Metrics to use for the query.
                query_type (str, optional): Type of query to perform. Defaults to 'vector'.

        Raises:
            ValueError: If the specified table is not found in the database.

        Returns:
            Any: PyArrow Table (Typing is unavailable due to Cython implementation)

        """
        if args.k is None:
            args.k = LanceDB.DEFAULT_K
        tbl = self._get_table(args.name)
        if tbl is None:
            msg = "Table not found in the database."
            raise ValueError(msg)

        filter_str = None
        if isinstance(args.filter_dict, dict):
            filter_str = " AND ".join(
                [f"{k} = '{v}'" for k, v in args.filter_dict.items()]
            )

        if args.metrics in ("L2", "cosine"):
            if filter_str:
                lance_query = (
                    tbl.search(query=query)
                    .limit(args.k)
                    .metric(args.metrics)
                    .where(filter_str, prefilter=args.prefilter)
                )
            else:
                lance_query = tbl.search(query=query).limit(args.k).metric(args.metrics)
        elif filter_str is not None:
            lance_query = (
                tbl.search(query=query)
                .limit(args.k)
                .where(filter_str, prefilter=args.prefilter)
            )
        else:
            lance_query = tbl.search(query=query).limit(args.k)
        if args.query_type == "hybrid" and self._reranker is not None:
            lance_query.rerank(reranker=self._reranker)

        docs = lance_query.to_arrow()
        if len(docs) == 0:
            warnings.warn("No results found for the query.", stacklevel=2)
        return docs

    def query(
        self,
        query: list[float] | str | tuple[list[float], str],
        args: LanceDBSearchArguments,
    ) -> LANCEDB_RESPONSE:
        res = self._query(query, args)
        return self._response_postprocess(res)
