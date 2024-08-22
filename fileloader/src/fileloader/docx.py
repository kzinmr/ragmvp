"""Docs parser.

Contains parsers for docx, pdf files.

"""

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem

from fileloader.base import BaseReader
from fileloader.schema import Document

logger = logging.getLogger(__name__)

RETRY_TIMES = 3


def xml2text(xml: bytes) -> str:
    """A string representing the textual content of this run, with content
    child elements like ``<w:tab/>`` translated to their Python
    equivalent.
    Adapted from: https://github.com/python-openxml/python-docx/
    """
    nsmap = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def _qn(tag: str) -> str:
        """Stands for 'qualified name', a utility function to turn a namespace
        prefixed tag name into a Clark-notation qualified tag name for lxml. For
        example, ``qn('p:cSld')`` returns ``'{http://schemas.../main}cSld'``.
        Source: https://github.com/python-openxml/python-docx/
        """
        prefix, tagroot = tag.split(":")
        uri = nsmap[prefix]
        return f"{{{uri}}}{tagroot}"

    text = ""
    root = ET.fromstring(xml)
    for child in root.iter():
        if child.tag == _qn("w:t"):
            t_text = child.text
            text += t_text if t_text is not None else ""
        elif child.tag == _qn("w:tab"):
            text += "\t"
        elif child.tag in (_qn("w:br"), _qn("w:cr")):
            text += "\n"
        elif child.tag == _qn("w:p"):
            text += "\n\n"
    return text


def docx2txt_process(docx: Path) -> str:
    """docx2txt: Extract text from a docx file.
    The code is taken and adapted from python-docx.
    It can however also extract text from header, footer and hyperlinks.
    """
    text = ""

    # unzip the docx in memory
    zipf = zipfile.ZipFile(docx)
    filelist = zipf.namelist()

    # get header text
    # there can be 3 header files in the zip
    header_xmls = "word/header[0-9]*.xml"
    for fname in filelist:
        if re.match(header_xmls, fname):
            text += xml2text(zipf.read(fname))

    # get main text
    doc_xml = "word/document.xml"
    text += xml2text(zipf.read(doc_xml))

    # get footer text
    # there can be 3 footer files in the zip
    footer_xmls = "word/footer[0-9]*.xml"
    for fname in filelist:
        if re.match(footer_xmls, fname):
            text += xml2text(zipf.read(fname))

    # if img_dir is not None:
    #     # extract images
    #     for fname in filelist:
    #         _, extension = os.path.splitext(fname)
    #         if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
    #             dst_fname = os.path.join(img_dir, os.path.basename(fname))
    #             with open(dst_fname, "wb") as dst_f:
    #                 dst_f.write(zipf.read(fname))

    zipf.close()
    return text.strip()


class DocxReader(BaseReader):
    """Docx parser."""

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> list[Document]:
        """Parse file."""
        if fs:
            with fs.open(str(file)) as f:
                text = docx2txt_process(f)
        else:
            text = docx2txt_process(file)

        if extra_info is None:
            extra_info = {"file_name": file.name}
        else:
            extra_info["file_name"] = file.name

        return [Document(text=text, metadata=extra_info)]
