import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TypeVar

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from pydantic import BaseModel
from tqdm import tqdm

from searchapi.config import get_settings
from searchapi.generator import get_generator
from searchapi.lancedb_wrapper import LANCEDB_QUERY_TYPE
from searchapi.retriever import SEARCH_TYPE, LanceDBVectorStore

T = TypeVar("T")


def batch(iterable: list[T], n: int = 1) -> Generator[list[T], None, None]:
    size = len(iterable)
    for ndx in range(0, size, n):
        yield iterable[ndx : min(ndx + n, size)]


def build_retriever(
    db_url: str,
    table_name: str,
    documents: list[Document],
    search_type: SEARCH_TYPE = "similarity",
    query_type: LANCEDB_QUERY_TYPE = "hybrid",
    k: int = 5,
    n_batches: int = 10,
) -> Runnable[str, list[Document]]:
    docsearch = LanceDBVectorStore(
        uri=db_url,
        table_name=table_name,
    )

    for doc in documents:
        doc.id = str(uuid.uuid4())

    # pricing tier上げないと429 Rate Limit エラー
    # ids = [d.id for d in documents]
    # texts = [d.page_content for d in documents]
    # metadatas = [d.metadata for d in documents]
    # docsearch.lancedb.add_texts(texts, metadatas, ids)

    batch_size = 2  # token数が均等になるように調整したい
    while len(documents) / batch_size > n_batches:
        batch_size *= 2
    print(f"Ingesting {len(documents)} Documents in {batch_size} batches...")

    total = round(len(documents) / batch_size)
    for doc_batch in tqdm(batch(documents, batch_size), total=total):
        ids = [d.id for d in doc_batch]
        texts = [d.page_content for d in doc_batch]
        metadatas = [d.metadata for d in doc_batch]
        # NOTE: document.page_content からの埋め込みしかできてない
        docsearch.lancedb.add_texts(texts, metadatas, ids)

    # KMeans: can not train num_partitions centroids with number of vectors, choose a smaller K instead
    # num_partitions = 256
    # while len(documents) < num_partitions:
    #     num_partitions //= 2
    # docsearch.create_index(
    #     num_partitions=num_partitions,
    #     num_sub_vectors=96,
    #     metric="l2",
    # )
    # docsearch.create_index(col_name="id")
    docsearch.create_fts_index()

    return docsearch.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "query_type": query_type, "filter_dict": {}},
    )


class RAGRunnable:
    def __init__(
        self,
        documents: list[Document],
        retriever: Runnable[str, list[Document]] | None = None,
        generetor: Runnable[str, BaseModel] | None = None,
    ) -> None:
        if retriever is None:
            db_url = get_settings().lancedb_dir
            table_name = get_settings().lancedb_table
            if Path(db_url).exists():
                shutil.rmtree(db_url)
            retriever = build_retriever(
                db_url=db_url,
                table_name=table_name,
                documents=documents,
                search_type="similarity",
                query_type="hybrid",
                k=5,
            )
        self.retriever = retriever  # BaseRetriever から緩和
        self.retriever_runnable = (lambda x: x["input"]) | self.retriever

        if generetor is None:
            generetor = get_generator()
        self.generetor_runnable = generetor

        # runnable.invoke({"input": "What is xxx?"})`
        # "input" は retriever_runnable にも generator_runnable にも渡される
        self.runnable: Runnable[str, dict | BaseModel] = RunnablePassthrough.assign(
            documents=self.retriever_runnable
        ).assign(answer=self.generetor_runnable)


def get_rag_chain(documents: list[Document]) -> Runnable[str, dict | BaseModel]:
    return RAGRunnable(documents).runnable
