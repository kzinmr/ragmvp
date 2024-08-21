import json
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document as LCDocument


def save_docs_to_jsonl(array: Iterable[LCDocument], file_path: Path) -> None:
    with file_path.open("w", encoding="utf8") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json(ensure_ascii=False) + "\n")


def load_docs_from_jsonl(file_path: Path) -> list[LCDocument]:
    array = []
    with file_path.open() as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = LCDocument(**data)
            array.append(obj)
    return array
