"""Tabular parser.

Contains parsers for tabular data files.

"""

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
from fsspec import AbstractFileSystem

from fileloader.base import BaseReader
from fileloader.schema import Document


class PandasExcelReader(BaseReader):
    """Pandas-based Excel parser.

    Parses Excel files using the Pandas `read_excel`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        sheet_name (str | int | None): Defaults to None, for all sheets, otherwise pass a str or int to specify the sheet to read.

        pandas_config (dict): Options for the `pandas.read_excel` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            for more information.
            Set to empty dict by default.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        sheet_name: str | None = None,
        pandas_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        if pandas_config is None:
            pandas_config = {}
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Parse .xlsx file."""
        openpyxl_spec = importlib.util.find_spec("openpyxl")
        if openpyxl_spec is not None:
            pass
        else:
            msg = "Please install openpyxl to read Excel files. You can install it with 'pip install openpyxl'"
            raise ImportError(msg)

        if extra_info is None:
            extra_info = {}

        # sheet_name of None is all sheets, otherwise indexing starts at 0
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        documents = []

        extra_info["file_path"] = str(file)

        # handle the case where only a single DataFrame is returned
        if isinstance(dfs, pd.DataFrame):
            extra_info["total_pages"] = 1
            extra_info["page_number"] = 1

            # Convert DataFrame to list of rows
            text_list = (
                dfs.fillna("")
                .astype(str)
                .apply(lambda row: " ".join(row.values), axis=1)
                .tolist()
            )

            if self._concat_rows:
                documents.append(
                    Document(text="\n".join(text_list), metadata=extra_info)
                )
            else:
                documents.extend(
                    [Document(text=text, metadata=extra_info) for text in text_list]
                )
        else:
            dfs_list = list(dfs.values())
            extra_info["total_pages"] = len(dfs_list)

            for i, _df in enumerate(dfs_list, 1):
                extra_info["page_number"] = i

                # Convert DataFrame to list of rows
                text_list = (
                    _df.fillna("")
                    .astype(str)
                    .apply(lambda row: " ".join(row), axis=1)
                    .tolist()
                )

                if self._concat_rows:
                    documents.append(
                        Document(text="\n".join(text_list), metadata=extra_info)
                    )
                else:
                    documents.extend(
                        [Document(text=text, metadata=extra_info) for text in text_list]
                    )

        return documents
