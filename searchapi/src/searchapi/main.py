import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents.base import Document

from searchapi.rag import get_rag_chain

load_dotenv()


def load_docs_from_jsonl(file_path: Path) -> list[Document]:
    array = []
    with file_path.open() as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


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

    rag_chain = get_rag_chain(documents)

    import time

    start = time.time()
    response = rag_chain.invoke({"input": "What is AI?"})
    elapsed_time = time.time() - start
    print(f"RAG in {elapsed_time:.4f} sec")
    print(response)

    # print(docsearch.get_table().schema)
