from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import BaseModel

from searchapi.config import get_settings

# LangChainで得ているもの:
# - ChatModel API: generate, stream, batch
# - Structured API: with_structured_output, bind_tools
# - LCEL API: invoke (Runnable)


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


# ... Runnable
# BaseLanguageModel(ABC) -> invoke, generate, stream, batch / with_structured_output, bind_tools
# BaseChatModel(ABC) -> invoke, generate, stream, batch / with_structured_output, bind_tools
# BaseChatOpenAI -> _generate, _stream / with_structured_output, bind_tools, bind_functions,
# AzureChatOpenAI -> with_structured_output, bind_tools


class GeneratorRunnable:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        if llm is None:
            llm = AzureChatOpenAI(
                azure_deployment=get_settings().azure_openai_chat_deployment,
                api_version=get_settings().azure_openai_chat_api_version,
            )
        self.llm = llm
        self.answer_runnable = self._generator_runnable(self.llm)

    def _generator_runnable(
        self, llm: BaseChatModel
    ) -> Runnable[dict[str, str | list[Document]], dict | BaseModel]:
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

        return (
            {
                "context": lambda x: self._stuff_docs(x["documents"]),
                "input": lambda x: x["input"],
            }
            | prompt_template
            | llm.with_structured_output(AnswerWithSources)
        )

    @staticmethod
    def _stuff_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)


def get_generator() -> Runnable[dict[str, str | list[Document]], dict | BaseModel]:
    return GeneratorRunnable().answer_runnable
