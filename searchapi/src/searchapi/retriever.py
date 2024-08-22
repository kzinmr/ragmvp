import logging
import math
import warnings
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol

import numpy as np
from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForRetrieverRun,
    CallbackManager,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    ensure_config,
)
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from typing_extensions import runtime_checkable

from searchapi.lancedb_wrapper import (
    LANCEDB_RESPONSE,
    LanceDB,
    LanceDBSearchArguments,
    embedding_func,
)

logger = logging.getLogger(__name__)

Matrix = list[list[float]] | list[np.ndarray] | np.ndarray
SEARCH_TYPE = Literal["similarity", "similarity_score_threshold", "mmr"]
SEARCH_RESPONSE = list[Document] | list[tuple[Document, float]]


class SearchArguments(LanceDBSearchArguments):
    """Search arguments for LanceDB search functions.
    - score: Whether to return relevance scores.
    - score_threshold: Minimum relevance threshold for similarity_score_threshold.
    - mmr_fetch_k: Number of Documents to fetch to pass to MMR algorithm.
    - mmr_lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
    LanceDBSearchArguments:
    # - k: Number of documents to return.
    # - name: Name of the table to search in.
    # - filter_dict: Optional filter arguments
    # - prefilter: Whether to apply the filter prior to the vector search.
    # - metrics: Metrics to use for the query.
    # - query_type: Type of query to perform in LanceDB.
    """

    score: bool = False
    score_threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    mmr_fetch_k: Annotated[int, Field(ge=1)] = 20
    mmr_lambda_mult: Annotated[float, Field(ge=0, le=1)] = 0.5
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class VectorStoreLike(Protocol):
    def similarity_search(
        self, query: str, args: SearchArguments
    ) -> list[Document]: ...

    def similarity_search_with_relevance_scores(
        self, query: str, args: SearchArguments
    ) -> list[tuple[Document, float]]: ...

    def max_marginal_relevance_search(
        self, query: str, args: SearchArguments
    ) -> list[Document]: ...


def select_relevance_score_fn(
    distance: Literal["cosine", "l2", "ip"],
) -> Callable[[float], float]:
    """Select the distance conversion function for the correct normalized relevance score.
    The 'correct' relevance function may differ depending on a few things, including:
    - the distance / similarity metric used by the Vector DB
    - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    - embedding dimensionality
    - etc.
    """

    def _euclidean_relevance_score_fn(distance: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        return 1.0 - distance / math.sqrt(2)

    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 1.0 - distance

    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        if distance > 0:
            return 1.0 - distance

        return -1.0 * distance

    if distance == "cosine":
        return _cosine_relevance_score_fn
    if distance == "l2":
        return _euclidean_relevance_score_fn
    if distance == "ip":
        return _max_inner_product_relevance_score_fn
    msg = (
        f"No supported normalization function for distance metric of type: {distance}. "
    )
    raise ValueError(msg)


def cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:  # type: ignore[valid-type]
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(x) == 0 or len(y) == 0:
        return np.array([])

    _x = np.array(x)
    _y = np.array(y)
    if _x.shape[1] != _y.shape[1]:
        msg = f"Number of columns in X and Y must be the same. X has shape {_x.shape} and Y has shape {_y.shape}."
        raise ValueError(msg)
    try:
        import simsimd as simd

        _x = np.array(x, dtype=np.float32)
        _y = np.array(y, dtype=np.float32)
        return 1 - np.array(simd.cdist(_x, _y, metric="cosine"))
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(_x, axis=1)
        y_norm = np.linalg.norm(_y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(_x, _y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    mmr_lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                mmr_lambda_mult * query_score - (1 - mmr_lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class LanceDBVectorStore(VectorStoreLike):
    """`LanceDB` vector store.

    Args:
        uri: URI to use for the LanceDB connection.
        table_name: Name of the table to use.
        relevance_score_fn: Function to use for calculating relevance scores.

    """

    def __init__(
        self,
        table_name: str | None = None,
        uri: str | None = None,
    ) -> None:
        """Initialize with Lance DB vectorstore"""
        self.lancedb = LanceDB(
            uri=uri,
            table_name=table_name,
        )

    def _response_postprocess(
        self, results: LANCEDB_RESPONSE, score: bool = False
    ) -> SEARCH_RESPONSE:
        """Convert LanceDB wrapped rensponses to LangChain Documents."""
        if not score:
            return [
                Document(
                    page_content=result.page_content,
                    metadata=result.metadata,
                )
                for result, _ in results
            ]

        return [
            (
                Document(
                    page_content=result.page_content,
                    metadata=result.metadata,
                ),
                _score,
            )
            for result, _score in results
        ]

    def _query(
        self,
        query: list[float] | str | tuple[list[float], str],
        args: LanceDBSearchArguments,
    ) -> LANCEDB_RESPONSE:
        """LanceDB query function.

        Args:
            query (list[float] | tuple[list[float], str]): Query to search for.
            args (LanceDBSearchArguments): Search arguments in LanceDB:
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
        return self.lancedb.query(query, args)

    def create_vector_index(
        self,
        num_partitions: int | None = 256,
        num_sub_vectors: int | None = 96,
        index_cache_size: int | None = None,
        metric: str | None = "L2",
        table_name: str | None = None,
    ) -> None:
        """Create a vector index on a table.
        Make sure your vector column has enough data before creating an index on it.

        Args:
            metric: Provide the metric to use for vector index. Defaults to 'L2'
                    choice of metrics: 'L2', 'dot', 'cosine'
            num_partitions: Number of partitions to use for the index. Defaults to 256.
            num_sub_vectors: Number of sub-vectors to use for the index. Defaults to 96.
            index_cache_size: Size of the index cache. Defaults to None.
            table_name: Name of the table to create index on. Defaults to None.

        Returns:
            None

        """
        self.lancedb.create_vector_index(
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            index_cache_size=index_cache_size,
            metric=metric,
            table_name=table_name,
        )

    def create_scalar_index(
        self,
        col_name: str,
        table_name: str | None = None,
    ) -> None:
        """Create a scalar(for non-vector cols) on a table.

        Args:
            col_name: Provide if you want to create index on a non-vector column.
            table_name: Name of the table to create index on. Defaults to None.

        Returns:
            None

        """
        self.lancedb.create_scalar_index(col_name, table_name=table_name)

    def create_fts_index(
        self,
        field_names: list[str] | None = None,
        table_name: str | None = None,
    ) -> None:
        self.lancedb.create_fts_index(field_names=field_names, table_name=table_name)

    def similarity_search_by_text(
        self,
        query: str,
        args: SearchArguments,
    ) -> SEARCH_RESPONSE:
        """Return documents most similar to the query considering the text surface."""
        if args.query_type not in ("fts", "hybrid"):
            msg = "This function is only supported for FTS and Hybrid search."
            raise NotImplementedError(msg)

        if not self.lancedb.is_fts_indexed:
            self.create_fts_index(table_name=args.name)

        if args.query_type == "hybrid":
            embedding = embedding_func.compute_source_embeddings(query)
            res = self._query(
                (embedding[0], query), LanceDBSearchArguments(**args.model_dump())
            )
        else:
            res = self._query(query, LanceDBSearchArguments(**args))

        return self._response_postprocess(res, score=args.score)

    def similarity_search_by_vector(
        self, embedding: list[float], args: SearchArguments
    ) -> SEARCH_RESPONSE:
        """Return documents most similar to the query vector."""
        res = self._query(embedding, LanceDBSearchArguments(**args))
        return self._response_postprocess(res, score=args.score)

    def similarity_search_with_score(
        self,
        query: str,
        args: SearchArguments,
    ) -> SEARCH_RESPONSE:
        """Return documents most similar to the query with relevance scores."""
        if args.query_type in ("fts", "hybrid"):
            return self.similarity_search_by_text(query, args)
        # query_type == "vector" or "auto"
        embedding = embedding_func.compute_source_embeddings(query)
        return self.similarity_search_by_vector(embedding[0], args)

    def similarity_search(
        self,
        query: str,
        args: SearchArguments,
    ) -> list[Document]:
        """Return documents most similar to the query.
        search_type == "similarity" mode in LanceDBRetriever

        Args:
            query: String to query the vectorstore with.
            args: search parameters; See `SearchArguments` for details.
        Raises:
            ValueError: If the specified table is not found in the database.

        Returns:
            list of documents most similar to the query.

        """
        args.score = False
        return self.similarity_search_with_score(query, args)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        args: SearchArguments,
    ) -> list[tuple[Document, float]]:
        """Similarity search with relevance scores.
        Return docs and relevance scores in the range [0, 1].
        0 is dissimilar, 1 is most similar.
        search_type == "similarity_score_threshold" mode in LanceDBRetriever.

        Args:
            query: Input text.
            args: search parameters; See `SearchArguments` for details.
                unique to this function:
                    score_threshold: a floating point value between 0 to 1
                                to filter the resulting set of retrieved docs.

        Returns:
            list of Tuples of (doc, similarity_score)

        """
        match args.metrics:
            case "cosine":
                relevance_score_fn = select_relevance_score_fn("cosine")
            case "L2":
                relevance_score_fn = select_relevance_score_fn("l2")
            # case "dot":  # not supported in LanceDB
            #     relevance_score_fn = select_relevance_score_fn("ip")
            case _:
                msg = "Unknown distance metric"
                raise ValueError(msg)

        args.score = True
        docs_and_scores = self.similarity_search_with_score(query, args)
        docs_and_similarities = [
            (doc, relevance_score_fn(float(score))) for doc, score in docs_and_scores
        ]

        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                f"Relevance scores must be between 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )

        if args.score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= args.score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    f"No relevant docs were retrieved using the relevance score threshold {args.score_threshold}",
                    stacklevel=2,
                )
        return docs_and_similarities

    def max_marginal_relevance_search(
        self,
        query: str,
        args: SearchArguments,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        search_type == "mmr" mode in LanceDBRetriever

        Args:
            query: Text to look up documents similar to.
            args: search parameters; See `SearchArguments` for details.
                unique to this function:
                    mmr_fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                    mmr_lambda_mult: Number between 0 and 1 that determines the degree
                                of diversity among the results with 0 corresponding
                                to maximum diversity and 1 to minimum diversity.

        Returns:
            list of Documents selected by maximal marginal relevance.

        """
        _embedding = embedding_func.compute_source_embeddings(query)
        embedding = _embedding[0]

        prefetch_args = LanceDBSearchArguments(**args)
        prefetch_args.k = args.mmr_fetch_k
        results = self._query(embedding, prefetch_args)
        embedding_list = [result.vector for result, _ in results]
        mmr_selected = maximal_marginal_relevance(
            query_embedding=embedding,  # np.array(embedding, dtype=np.float32),
            embedding_list=embedding_list,
            mmr_lambda_mult=args.mmr_lambda_mult,
            k=args.k,
        )

        candidates = self._response_postprocess(results, score=args.score)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def search(
        self,
        query: str,
        search_type: SEARCH_TYPE,
        search_kwargs: dict[str, Any],
    ) -> SEARCH_RESPONSE:
        """Return docs most similar to query using a specified search type.

        Args:
            query: Input text
            search_type: Type of search to perform. Can be
                    "similarity", "similarity_score_threshold" or "mmr".
            search_kwargs: search parameters; See. `SearchArguments`.

        Returns:
            list of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not one of
                "similarity", "similarity_score_threshold" or "mmr".

        """
        search_args = SearchArguments.model_validate(search_kwargs)
        match search_type:
            case "similarity":
                return self.similarity_search(query, search_args)
            case "similarity_score_threshold":
                return self.similarity_search_with_relevance_scores(query, search_args)
            case "mmr":
                return self.max_marginal_relevance_search(query, search_args)
            case _:
                msg = f"search_type of {search_type} not allowed. Expected search_type to be 'similarity', 'similarity_score_threshold' or 'mmr'."
                raise ValueError(msg)

    def _get_retriever_tags(self) -> list[str]:
        """Get tags for retriever."""
        return [self.__class__.__name__]
        # if self.embeddings:
        #     tags.append(self.embeddings.__class__.__name__)

    def as_retriever(
        self, search_type: SEARCH_TYPE, search_kwargs: dict[str, Any], **kwargs: Any
    ) -> "LanceDBRetriever":
        """Return LanceDBRetriever initialized from this VectorStore.

        Args:
            search_type (Optional[str]): Defines the type of search that the Retriever should perform.
                Can be "similarity" (default), "mmr", or "similarity_score_threshold".
            search_kwargs (Optional[dict]): Keyword arguments to pass to the search function.
                Can include things like:
                    k: Number of documents to return.
                    name: Name of the table to search in.
                    filter_dict (Optional[dict]): Filter by document metadata
                    prefilter(Optional[bool]): Whether to apply the filter prior to the vector search.
                    metrics: Metrics to use for the query.
                    query_type: Type of query to perform in LanceDB.
                    score_threshold: Minimum relevance threshold for similarity_score_threshold.
                    mmr_fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                    mmr_lambda_mult: Diversity of results returned by MMR;
                                    1 for minimum diversity and 0 for maximum. (Default: 0.5)
            **kwargs: Additional keyword arguments.

        Returns:
            LanceDBRetriever: Retriever class for VectorStore.

        """
        search_args = SearchArguments.model_validate(search_kwargs)
        tags = kwargs.pop("tags", None) or [*self._get_retriever_tags()]
        return LanceDBRetriever(
            vectorstore=self,
            search_type=search_type,
            search_kwargs=search_args,
            tags=tags,
            **kwargs,
        )


# RunnableSerializable[RetrieverInput, RetrieverOutput]
# Input_contra = TypeVar("Input_contra", contravariant=True)
# Output type should implement __concat__, as eg str, list, dict do
# Output_co = TypeVar("Output_co", covariant=True)
RetrieverInput = str
RetrieverOutput = list[Document]


class LanceDBRetriever(BaseModel, Runnable[RetrieverInput, RetrieverOutput]):
    """LanceDB Retriever class for VectorStore.
    Note that async retrieval is called via _aget_relevant_documents
    in BaseRetriever, which is implemented with ThreadPoolExecutor.
    This method is called in retriever.ainvoke().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    vectorstore: VectorStoreLike
    search_type: SEARCH_TYPE = "similarity"
    search_kwargs: SearchArguments
    _new_arg_supported: bool = True  # parameters.get("run_manager") is not None
    _expects_other_args: bool = False
    tags: list[str] | None = None
    """Tags to add to the run trace."""
    metadata: dict[str, Any] | None = None
    """Metadata to add to the run trace."""

    @classmethod
    @model_validator(mode="before")
    def validate_search_type(cls, values: ValidationInfo) -> dict:
        """Validate search type.

        Args:
            values: Values to validate.

        Returns:
            Values: Validated values.

        Raises:
            ValueError: If score_threshold is not specified with a float value(0~1)

        """
        search_type = values.data.get("search_type", "similarity")
        match search_type:
            case "similarity_score_threshold":
                score_threshold = values.data.get("search_kwargs", {}).get(
                    "score_threshold"
                )
                if score_threshold is None:
                    msg = "`score_threshold` is not specified with a float value(0~1) in `search_kwargs`."
                    raise ValueError(msg)
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # run_manager is called in caller method (.invoke())
        # relevance scores and queries can be logged via run_manager with extension.
        match self.search_type:
            case "similarity":
                self.search_kwargs.score = False
                return self.vectorstore.similarity_search(query, self.search_kwargs)
            case "similarity_score_threshold":
                self.search_kwargs.score = True
                docs_and_similarities = (
                    self.vectorstore.similarity_search_with_relevance_scores(
                        query, self.search_kwargs
                    )
                )
                return [doc for doc, _ in docs_and_similarities]
            case "mmr":
                self.search_kwargs.score = False
                return self.vectorstore.max_marginal_relevance_search(
                    query, self.search_kwargs
                )
            case _:
                msg = f"search_type of {self.search_type} not allowed."
                raise ValueError(msg)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents

        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )

    def invoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Invoke the retriever to get relevant documents.

        Main entry point for synchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:

        .. code-block:: python

            retriever.invoke("query")

        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            # **self._get_ls_params(**kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    input, run_manager=run_manager, **_kwargs
                )
            else:
                result = self._get_relevant_documents(input, **_kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e  # noqa: TRY201
        else:
            run_manager.on_retriever_end(
                result,
            )
            return result

    async def ainvoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously invoke the retriever to get relevant documents.

        Main entry point for asynchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:

        .. code-block:: python

            await retriever.ainvoke("query")

        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            # **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    input, run_manager=run_manager, **_kwargs
                )
            else:
                result = await self._aget_relevant_documents(input, **_kwargs)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e  # noqa: TRY201
        else:
            await run_manager.on_retriever_end(
                result,
            )
            return result
