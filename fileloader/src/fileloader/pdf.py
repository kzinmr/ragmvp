"""Read PDF files using PyMuPDF library."""

from pathlib import Path
from typing import Any

import fitz

from fileloader.base import BaseReader
from fileloader.schema import Document


class PyMuPDFReader(BaseReader):
    """Read PDF files using PyMuPDF library."""

    def load_data(
        self,
        file_path: Path | str,
        extra_info: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.
        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            extra_info (Optional[dict], optional): extra information related to each document in dict format. Defaults to None.
        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.
        Returns:
            list[Document]: list of documents.

        """
        # open PDF file
        doc = fitz.open(file_path)

        if extra_info is None:
            extra_info = {}
        extra_info["total_pages"] = len(doc)
        extra_info["file_path"] = str(file_path)

        return [
            Document(
                text=page.get_text().encode("utf-8"),
                metadata=extra_info | {"page_number": page.number + 1},
            )
            for page in doc
        ]
