"""Simple reader that reads files of different formats from a directory."""

import asyncio
import logging
import mimetypes
import multiprocessing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from functools import reduce
from itertools import repeat
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec
from fsspec.implementations.local import LocalFileSystem
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from fileloader.base import BaseReader, ResourcesReaderMixin
from fileloader.docx import DocxReader
from fileloader.pdf import PyMuPDFReader
from fileloader.schema import Document
from fileloader.slide import PptxReader
from fileloader.tabular import PandasExcelReader

PathLike = Path | PurePosixPath


def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".pdf": PyMuPDFReader,  # PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".xls": PandasExcelReader,
        ".xlsx": PandasExcelReader,
        # ".hwp": HWPReader,
        # ".gif": ImageReader,
        # ".jpg": ImageReader,
        # ".png": ImageReader,
        # ".jpeg": ImageReader,
        # ".webp": ImageReader,
        # ".mp3": VideoAudioReader,
        # ".mp4": VideoAudioReader,
        # ".csv": PandasCSVReader,
        # ".epub": EpubReader,
        # ".md": MarkdownReader,
        # ".mbox": MboxReader,
        # ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


class FileSystemReaderMixin(ABC):
    @abstractmethod
    def read_file_content(self, input_file: Path, **kwargs: Any) -> bytes:
        """Read the bytes content of a file.

        Args:
            input_file (Path): Path to the file.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bytes: File content.

        """

    async def aread_file_content(self, input_file: Path, **kwargs: Any) -> bytes:
        """Read the bytes content of a file asynchronously.

        Args:
            input_file (Path): Path to the file.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bytes: File content.

        """
        return self.read_file_content(input_file, **kwargs)


def _format_file_timestamp(timestamp: float) -> str | None:
    """Format file timestamp to a %Y-%m-%d string.

    Args:
        timestamp (float): timestamp in float

    Returns:
        str: formatted timestamp

    """
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def default_file_metadata_func(
    file_path: str, fs: fsspec.AbstractFileSystem | None = None
) -> dict[str, Any]:
    """Get some handy metadata from filesystem.

    Args:
        file_path: str: file path in str
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
            in the constructor, it will override the fs parameter here.

    """
    fs = fs or get_default_fs()
    stat_result = fs.stat(file_path)

    try:
        file_name = Path(str(stat_result["name"])).name
    except Exception:
        file_name = Path(file_path).name

    creation_date = _format_file_timestamp(stat_result.get("created"))
    last_modified_date = _format_file_timestamp(stat_result.get("mtime"))
    last_accessed_date = _format_file_timestamp(stat_result.get("atime"))
    default_meta = {
        "file_path": file_path,
        "file_name": file_name,
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": stat_result.get("size"),
        "creation_date": creation_date,
        "last_modified_date": last_modified_date,
        "last_accessed_date": last_accessed_date,
    }

    # Return not null value
    return {
        meta_key: meta_value
        for meta_key, meta_value in default_meta.items()
        if meta_value is not None
    }


class _DefaultFileMetadataFunc:
    """Default file metadata function wrapper which stores the fs.
    Allows for pickling of the function.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem | None = None) -> None:
        self.fs = fs or get_default_fs()

    def __call__(self, file_path: str) -> dict[str, Any]:
        return default_file_metadata_func(file_path, self.fs)


def get_default_fs() -> fsspec.AbstractFileSystem:
    return LocalFileSystem()


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


logger = logging.getLogger(__name__)


class SimpleDirectoryReader(BaseReader, ResourcesReaderMixin, FileSystemReaderMixin):
    """Simple directory reader.

    Load files from file directory.
    Automatically select the best file reader given file extensions.

    Args:
        input_dir (str): Path to the directory.
        input_files (list): list of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (list): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[list[str]]): list of required extensions.
            Default is None.
        file_extractor (Optional[dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, dict]]): A function that takes
            in a filename and returns a dict of metadata for the Document.
            Default is None.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
        to using the local file system. Can be changed to use any remote file system
        exposed via the fsspec interface.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    supported_suffix_fn: Callable[..., dict[str, type[BaseReader]]] = (
        _try_loading_included_file_formats
    )

    def __init__(
        self,
        input_dir: str | None = None,
        input_files: list[str] | None = None,
        exclude: list[str] | None = None,
        errors: str = "ignore",
        encoding: str = "utf-8",
        required_exts: list[str] | None = None,
        file_extractor: dict[str, BaseReader] | None = None,
        num_files_limit: int | None = None,
        file_metadata: Callable[[str], dict[str, Any]] | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
        recursive: bool = False,
        exclude_hidden: bool = True,
        show_progress: bool = False,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()

        if not input_dir and not input_files:
            msg = "Must provide either `input_dir` or `input_files`."
            raise ValueError(msg)

        self.fs = fs or get_default_fs()
        self.errors = errors
        self.encoding = encoding
        self.exclude = exclude

        self.show_progress = show_progress
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden

        self.required_exts = required_exts
        self.num_files_limit = num_files_limit
        _path = Path if is_default_fs(self.fs) else PurePosixPath

        if input_files:
            self.input_files = []
            for path in input_files:
                if not self.fs.isfile(path):
                    msg = f"File {path} does not exist."
                    raise ValueError(msg)
                input_file = _path(path)
                self.input_files.append(input_file)
        elif input_dir:
            if not self.fs.isdir(input_dir):
                msg = f"Directory {input_dir} does not exist."
                raise ValueError(msg)
            self.input_dir = _path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

        self.file_extractor = {}
        if file_extractor is not None:
            self.file_extractor = file_extractor

        self.file_metadata = file_metadata or _DefaultFileMetadataFunc(self.fs)

    def is_hidden(self, path: PathLike) -> bool:
        return any(
            part.startswith(".") and part not in [".", ".."] for part in path.parts
        )

    def _add_files(self, input_dir: PathLike) -> list[PathLike]:
        """Add files."""
        all_files: set[PathLike] = set()
        rejected_files: set[PathLike] = set()
        rejected_dirs: set[PathLike] = set()
        # Default to POSIX paths for non-default file systems (e.g. S3)
        _path: type[PathLike] = Path if is_default_fs(self.fs) else PurePosixPath

        if self.exclude is not None:
            self._collect_excluded_files_and_dirs(
                input_dir, _path, rejected_files, rejected_dirs
            )

        file_refs = self._get_file_refs(input_dir)

        for _ref in file_refs:
            ref = _path(_ref)
            is_dir = self.fs.isdir(ref)
            skip_because_hidden = self.exclude_hidden and self.is_hidden(ref)
            skip_because_bad_ext = self._is_bad_extension(ref)
            skip_because_excluded = self._is_excluded(
                ref, rejected_files, rejected_dirs
            )

            if not (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
            ):
                all_files.add(ref)

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            msg = f"No files found in {input_dir}."
            raise ValueError(msg)

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        logger.debug(
            "> [SimpleDirectoryReader] Total files added: {%s}", len(new_input_files)
        )

        return new_input_files

    def _collect_excluded_files_and_dirs(
        self,
        input_dir: PathLike,
        _path: type[PathLike],
        rejected_files: set[PathLike],
        rejected_dirs: set[PathLike],
    ) -> None:
        if self.exclude is not None:
            for excluded_pattern in self.exclude:
                if self.recursive:
                    # Recursive glob
                    excluded_glob = _path(input_dir) / _path("**") / excluded_pattern
                else:
                    # Non-recursive glob
                    excluded_glob = _path(input_dir) / excluded_pattern
                for file in self.fs.glob(str(excluded_glob)):
                    if self.fs.isdir(file):
                        rejected_dirs.add(_path(file))
                    else:
                        rejected_files.add(_path(file))

    def _get_file_refs(self, input_dir: PathLike) -> Any:
        if self.recursive:
            return self.fs.glob(str(input_dir) + "/**/*")
        return self.fs.glob(str(input_dir) + "/*")

    def _is_bad_extension(self, ref: PathLike) -> bool:
        return self.required_exts is not None and ref.suffix not in self.required_exts

    def _is_excluded(
        self, ref: PathLike, rejected_files: set[PathLike], rejected_dirs: set[PathLike]
    ) -> bool:
        if ref in rejected_files:
            return True
        ref_parent_dir = ref if self.fs.isdir(ref) else self.fs._parent(ref)
        for rejected_dir in rejected_dirs:
            if str(ref_parent_dir).startswith(str(rejected_dir)):
                logger.debug(
                    "Skipping %s because it in parent dir %s which is in %s",
                    ref,
                    ref_parent_dir,
                    rejected_dir,
                )
                return True
        return False

    def _exclude_metadata(self, documents: list[Document]) -> list[Document]:
        """Exclude metadata from documents.
        Args:
            documents (list[Document]): list of documents.

        """
        for doc in documents:
            # Keep only metadata['file_path'] in both embedding and llm content
            # str, which contain extreme important context that about the chunks.
            # Dates is provided for convenience of postprocessor such as
            # TimeWeightedPostprocessor, but excluded for embedding and LLMprompts
            doc.excluded_embed_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )
            doc.excluded_llm_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )

        return documents

    def list_resources(self, *args: Any, **kwargs: Any) -> list[str]:
        """list files in the given filesystem."""
        return [str(f) for f in self.input_files]

    def get_resource_info(self, resource_id: str) -> dict[str, Any]:
        info_result = self.fs.info(resource_id)

        creation_date = _format_file_timestamp(info_result.get("created"))
        last_modified_date = _format_file_timestamp(info_result.get("mtime"))

        info_dict = {
            "file_path": resource_id,
            "file_size": info_result.get("size"),
            "creation_date": creation_date,
            "last_modified_date": last_modified_date,
        }

        # Ignore None values
        return {
            meta_key: meta_value
            for meta_key, meta_value in info_dict.items()
            if meta_value is not None
        }

    def load_resource(
        self, resource_id: str, **kwargs: dict[str, Any]
    ) -> list[Document]:
        file_extractor = kwargs.get("file_extractor", self.file_extractor)
        _file_metadata = kwargs.get("file_metadata", self.file_metadata)
        file_metadata = None
        if not isinstance(_file_metadata, dict):
            file_metadata = _file_metadata

        fs = kwargs.get("fs", self.fs)

        path_func = Path if is_default_fs(fs) else PurePosixPath

        return SimpleDirectoryReader.load_file(
            input_file=path_func(resource_id),
            file_extractor=file_extractor,
            file_metadata=file_metadata,
            fs=fs,
            **kwargs,
        )

    async def aload_resource(
        self, resource_id: str, **kwargs: dict[str, Any]
    ) -> list[Document]:
        return await self.aload_file(input_file=Path(resource_id))

    def read_file_content(self, input_file: Path, **kwargs: Any) -> bytes:
        """Read file content."""
        fs: fsspec.AbstractFileSystem = kwargs.get("fs", self.fs)
        with fs.open(input_file, errors=self.errors, encoding=self.encoding) as f:
            return bytes(f.read())

    @staticmethod
    def load_file(
        input_file: Path,
        file_extractor: dict[str, BaseReader],
        file_metadata: Callable[[str], dict[str, Any]] | None,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Static method for loading file.

        NOTE: necessarily as a static method for parallel processing.

        Args:
            input_file (Path): File path to read
            file_metadata (Callable[[str], dict]): A function that takes
                in a filename and returns a dict of metadata for the Document.
            file_extractor (dict[str, BaseReader]): A mapping of file
                extension to a BaseReader class that specifies how to convert that file
                to text.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
                to using the local file system. Can be changed to use any remote file system

        Returns:
            list[Document]: loaded documents

        """
        default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
        metadata = file_metadata(str(input_file)) if file_metadata else None
        file_suffix = input_file.suffix.lower()
        if file_suffix in default_file_reader_cls or file_suffix in file_extractor:
            default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
            reader = SimpleDirectoryReader._get_file_reader(
                file_extractor, file_suffix, default_file_reader_cls
            )
            return SimpleDirectoryReader._load_with_reader(
                input_file,
                reader,
                metadata,
                fs,
            )

        return SimpleDirectoryReader._load_without_reader(
            input_file,
            metadata,
            fs,
        )

    async def aload_file(self, input_file: Path) -> list[Document]:
        """Load file asynchronously."""
        default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())
        metadata: dict[str, Any] | None = None
        documents: list[Document] = []

        if self.file_metadata is not None:
            metadata = self.file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()
        if (
            file_suffix in default_file_reader_suffix
            or file_suffix in self.file_extractor
        ):
            reader = SimpleDirectoryReader._get_file_reader(
                self.file_extractor, file_suffix, default_file_reader_cls
            )
            documents = await self._aload_documents_with_reader(
                input_file,
                reader,
                metadata,
                self.fs,
            )
        else:
            documents = self._load_without_reader(
                input_file, metadata, self.fs, self.encoding, self.errors
            )

        return documents

    @staticmethod
    def _get_file_reader(
        file_extractor: dict[str, BaseReader],
        file_suffix: str,
        default_file_reader_cls: dict[str, type[BaseReader]],
    ) -> BaseReader:
        if file_suffix not in file_extractor:
            # instantiate file reader if not already
            reader_cls = default_file_reader_cls[file_suffix]
            file_extractor[file_suffix] = reader_cls()
        return file_extractor[file_suffix]

    @staticmethod
    def _load_with_reader(
        input_file: Path,
        reader: BaseReader,
        metadata: dict[str, Any] | None,
        fs: fsspec.AbstractFileSystem | None,
    ) -> list[Document]:
        try:
            kwargs = {"extra_info": metadata}
            if fs and not is_default_fs(fs):
                kwargs["fs"] = fs
            docs = reader.load_data(input_file, **kwargs)
        except ImportError as e:
            raise ImportError(str(e)) from e
        except Exception as e:
            print(
                f"Failed to load file {input_file} with error: {e}. Skipping...",
                flush=True,
            )
            return []

        return docs

    @staticmethod
    async def _aload_documents_with_reader(
        input_file: Path,
        reader: BaseReader,
        metadata: dict[str, Any] | None,
        fs: fsspec.AbstractFileSystem | None,
    ) -> list[Document]:
        # NOTE: catch all errors except for ImportError
        try:
            kwargs = {"extra_info": metadata}
            if fs and not is_default_fs(fs):
                kwargs["fs"] = fs
            docs = await reader.aload_data(input_file, **kwargs)
        except ImportError as e:
            raise ImportError(str(e)) from e
        except Exception as e:
            print(
                f"Failed to load file {input_file} with error: {e}. Skipping...",
                flush=True,
            )
            return []

        return docs

    @staticmethod
    def _load_without_reader(
        input_file: Path,
        metadata: dict[str, Any] | None,
        fs: fsspec.AbstractFileSystem | None,
        encoding: str = "utf-8",
        errors: str = "ignore",
    ) -> list[Document]:
        fs = fs or get_default_fs()
        with fs.open(input_file, errors=errors, encoding=encoding) as f:
            data = f.read().decode(encoding, errors=errors)

        doc = Document(text=data, metadata=metadata or {})

        return [doc]

    def load_data(
        self,
        num_workers: int | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Load data from the input directory.
        Args:
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.
        Returns:
            list[Document]: A list of documents.

        """
        documents = []

        files_to_process = self.input_files
        fs = fs or self.fs

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count.",
                    stacklevel=2,
                )
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(
                    SimpleDirectoryReader.load_file,
                    zip(
                        files_to_process,
                        repeat(self.file_metadata),
                        repeat(self.file_extractor),
                        repeat(self.encoding),
                        repeat(self.errors),
                        repeat(fs),
                    ),
                )
                documents = reduce(lambda x, y: x + y, results)

        else:
            if self.show_progress:
                files_to_process = tqdm(
                    self.input_files, desc="Loading files", unit="file"
                )
            for input_file in files_to_process:
                documents.extend(
                    SimpleDirectoryReader.load_file(
                        input_file=input_file,
                        file_metadata=self.file_metadata,
                        file_extractor=self.file_extractor,
                        fs=fs,
                    )
                )

        return self._exclude_metadata(documents)

    async def aload_data(
        self,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Load data from the input directory.
        Args:
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.
        Returns:
            list[Document]: A list of documents.

        """
        files_to_process = self.input_files
        fs = fs or self.fs

        coroutines = [self.aload_file(input_file) for input_file in files_to_process]
        # NOTE: `num_workers`is dropped to avoid additional dependencies.
        # num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
        # if num_workers:
        #     document_lists = await run_jobs(
        #         coroutines, show_progress=self.show_progress, workers=num_workers
        #     )
        if self.show_progress:
            document_lists = await tqdm_asyncio.gather(*coroutines)
        else:
            document_lists = await asyncio.gather(*coroutines)
        documents = [doc for doc_list in document_lists for doc in doc_list]

        return self._exclude_metadata(documents)

    def iter_data(self) -> Generator[list[Document], Any, Any]:
        """Load data iteratively from the input directory.
        Returns:
            Generator[list[Document]]: A list of documents.

        """
        files_to_process = self.input_files

        if self.show_progress:
            files_to_process = tqdm(self.input_files, desc="Loading files", unit="file")

        for input_file in files_to_process:
            documents = SimpleDirectoryReader.load_file(
                input_file=input_file,
                file_metadata=self.file_metadata,
                file_extractor=self.file_extractor,
                fs=self.fs,
            )

            documents = self._exclude_metadata(documents)

            if len(documents) > 0:
                yield documents
