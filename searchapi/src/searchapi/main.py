import argparse
import json
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, TypedDict, TypeVar

from dotenv import load_dotenv
from langchain_core.documents.base import Document as LCDocument
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_openai.chat_models import AzureChatOpenAI
from tqdm import tqdm

from searchapi.config import get_settings
from searchapi.lancedb_wrapper import LANCEDB_QUERY_TYPE
from searchapi.retriever import SEARCH_TYPE, LanceDBVectorStore

load_dotenv()


def load_docs_from_jsonl(file_path: Path) -> list[LCDocument]:
    array = []
    with file_path.open() as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = LCDocument(**data)
            array.append(obj)
    return array


def rag_chain_lcel(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    # single-turn QA
    # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
    # prompt_template = hub.pull("rlm/rag-prompt")
    # multi-turn QA
    # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    # prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def _stuff_docs(docs: list[LCDocument]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # NOTE: 回答引用元生成は、LLM出力を関数呼び出しを通じて Pydantic Model に変換することで実現
    # 1. Pydantic Model定義をFunction Callingパラメタに変換し、JSON変換関数(tool)として登録:
    # - Pydantic model -> .model_json_schema() -> name/description/parameters へ変換
    # 2. JSONに変換されたLLMのレスポンスは呼び出し元にて値検証される:
    # - llm | langchain_core.output_parsers.openai_tools.PydanticToolsParser
    # See. https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
    class AnswerWithSources(TypedDict):
        """An answer to the question, with sources.
        Attributes:
            answer: The answer sentence to the question.
            sources: List of sources (filename and document page number) used to answer the question.

        """

        answer: str
        sources: Annotated[
            list[str],
            ...,
            "List of sources (filename and page number) used to answer the question",
        ]

    answer_from_docs = (
        {
            "context": lambda x: _stuff_docs(x["documents"]),
            "input": lambda x: x["input"],
        }
        | prompt_template
        | llm.with_structured_output(AnswerWithSources)
    )

    retrieve_docs = (lambda x: x["input"]) | retriever

    # return a RAG chain which can be invoked with a question like:
    # `response = rag_chain.invoke({"input": "What is xxx?"})`
    return RunnablePassthrough.assign(documents=retrieve_docs).assign(
        answer=answer_from_docs
    )


def build_retriever(
    db_url: str,
    table_name: str,
    documents: list[LCDocument],
    search_type: SEARCH_TYPE = "similarity",
    query_type: LANCEDB_QUERY_TYPE = "hybrid",
    k: int = 5,
    n_batches: int = 10,
) -> BaseRetriever:
    T = TypeVar("T")

    def batch(iterable: list[T], n: int = 1) -> Generator[list[T], None, None]:
        size = len(iterable)
        for ndx in range(0, size, n):
            yield iterable[ndx : min(ndx + n, size)]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bulk index database from the JSONL data")
    parser.add_argument(
        "--datadir",
        "-i",
        type=str,
        help="Path to the directory containing the data files",
    )

    args = vars(parser.parse_args())
    data_dir = args["datadir"]
    dataset_path = Path(data_dir) / "documents.jsonl"  # get_settings().dataset_path
    documents = load_docs_from_jsonl(dataset_path)

    # NOTE: Tableはスキーマレスに動的に定義される
    # langchain.Document 都合で metadata:dict フィールドの挿入が行われるがlancedbのschemaが辞書型に非対応
    db_url = get_settings().lancedb_dir
    table_name = get_settings().lancedb_table
    if Path(db_url).exists():
        shutil.rmtree(db_url)

    # embeddings = AzureOpenAIEmbeddings(
    #     azure_deployment=get_settings().azure_openai_embedding_deployment,
    #     api_version=get_settings().azure_openai_embedding_api_version,
    #     model="text-embedding-3-large",
    #     dimensions=get_settings().embedding_dimension,
    #     embedding_ctx_length=8191,
    #     # show_progress_bar=True,
    # )

    # 128K context window and 16K output window
    llm = AzureChatOpenAI(
        azure_deployment=get_settings().azure_openai_chat_deployment,
        api_version=get_settings().azure_openai_chat_api_version,
    )

    retriever = build_retriever(
        db_url=db_url,
        table_name=table_name,
        documents=documents,
        search_type="similarity",
        query_type="hybrid",
        k=5,
    )

    rag_chain = rag_chain_lcel(retriever, llm)

    import time

    start = time.time()
    response = rag_chain.invoke({"input": "What is AI?"})
    elapsed_time = time.time() - start
    print(f"RAG in {elapsed_time:.4f} sec")
    print(response)

    # print(docsearch.get_table().schema)
