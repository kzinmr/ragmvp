import argparse
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, TypedDict

import lancedb
from dotenv import load_dotenv
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from tqdm import tqdm

from config import get_settings
from fileloader.utils import load_docs_from_jsonl

load_dotenv()

def rag_chain_lcel(vectorstore: VectorStore, llm: BaseChatModel) -> Runnable:
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # call .similarity_search()
        search_kwargs={"k": 5, "query_type": "vector"},
    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bulk index database from the JSONL data")
    parser.add_argument(
        "--filename",
        type=str,
        default="documents.jsonl",
        help="Name of the JSONL file to use",
    )
    args = vars(parser.parse_args())
    filename = args["filename"]
    data_dir = Path(__file__).parents[1] / "data"
    documents = load_docs_from_jsonl(data_dir / filename)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=get_settings().azure_openai_embedding_deployment,
        api_version=get_settings().azure_openai_embedding_api_version,
        model="text-embedding-3-large",
        dimensions=get_settings().embedding_dimension,
        embedding_ctx_length=8191,
        # show_progress_bar=True,
    )
    # 128K context window and 16K output window
    llm = AzureChatOpenAI(
        azure_deployment=get_settings().azure_openai_chat_deployment,
        api_version=get_settings().azure_openai_chat_api_version,
    )

    # table, db_connection = open_table()
    db_url = get_settings().lancedb_dir
    if Path(db_url).exists():
        shutil.rmtree(db_url)
    db = lancedb.connect(db_url)

    # NOTE: Tableはスキーマレスに動的に定義される
    # NOTE: langchain 内で metadata:dict フィールドの挿入が行われるがlancedbのschemaが辞書型に非対応
    table_name = get_settings().lancedb_table
    # NOTE: id はlancedb(langchain)の中でuuidがふられる
    # NOTE: document.page_content からの埋め込みしかできない

    def batch(iterable: list, n: int = 1) -> Generator[list, None, None]:
        size = len(iterable)
        for ndx in range(0, size, n):
            yield iterable[ndx : min(ndx + n, size)]

    batch_size = 2
    NUM_BATCHES = 10
    while len(documents) / batch_size > NUM_BATCHES:
        batch_size *= 2
    print(f"Ingesting {len(documents)} Documents in {batch_size} batches...")
    total = round(len(documents) / batch_size)
    for doc_batch in tqdm(batch(documents, batch_size), total=total):
        docsearch = LanceDB.from_documents(
            doc_batch, embeddings, connection=db, table_name=table_name
        )
    # index_chunks(table, chunked_data)

    rag_chain = rag_chain_lcel(docsearch, llm)

    import time

    start = time.time()
    response = rag_chain.invoke({"input": "What is adsai?"})
    elapsed_time = time.time() - start
    print(f"RAG in {elapsed_time:.4f} sec")
    print(response)

    # print(docsearch.get_table().schema)
