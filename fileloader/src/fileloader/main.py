import argparse
from pathlib import Path

from fileloader.file import SimpleDirectoryReader
from fileloader.langchain import LCDocument, RecursiveCharacterTextSplitter
from fileloader.utils import save_docs_to_jsonl


def load_data_from_directory(data_dir: Path, limit: int) -> list[LCDocument]:
    num_files_limit = limit if limit > 0 else None
    documents = SimpleDirectoryReader(
        str(data_dir),
        recursive=True,
        num_files_limit=num_files_limit,
        required_exts=[".pptx", ".ppt", ".pptm", ".pdf", ".xlsx", ".xls", ".docx"],
        show_progress=True,
    ).load_data()

    return [doc.to_langchain_format() for doc in documents]


def chunk_documents(
    documents: list[LCDocument], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[LCDocument]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    lc_docs = text_splitter.split_documents(documents)  # type: ignore[no-any-return]
    return [
        LCDocument(page_content=doc.page_content, metadata=doc.metadata)
        for doc in lc_docs
    ]


def filter_short_documents(
    documents: list[LCDocument], min_length: int = 100
) -> list[LCDocument]:
    return [doc for doc in documents if len(doc.page_content) > min_length]


def load_main(
    data_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200, limit: int = 0
) -> list[LCDocument]:
    """Document(
        id="chunk-2",
        page_content="Hello, world!",
        metadata={"source": "doc-1"}
    )
    """
    documents = load_data_from_directory(data_dir, limit)

    documents = chunk_documents(documents, chunk_size, chunk_overlap)
    return filter_short_documents(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Bulk conversion from document files to JSONL dataset."
    )
    parser.add_argument(
        "--datadir",
        "-i",
        type=str,
        help="Path to the directory containing the data files",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        help="Path to the directory to save the JSONL dataset",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit the size of the dataset to load for testing purposes",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Size of each chunk to break the dataset into before processing",
    )
    args = vars(parser.parse_args())

    limit = args["limit"]
    chunksize = args["chunksize"]
    data_dir = Path(args["datadir"])
    out_dir = Path(args["outdir"])

    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    documents = load_main(data_dir, chunk_size=chunksize, limit=limit)

    out_path = data_dir / "documents.jsonl"
    save_docs_to_jsonl(documents, out_path)
