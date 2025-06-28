import logging
import os
import pathlib
import tempfile
from typing import Any, List, Union

from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from streamlit.logger import get_logger

logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = get_logger(__name__)

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: Union[str,List[str]], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    """Loads in a document with a supported extention"""

    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

def load_document(temp_filepath: str) -> list[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(f"Invalid extension type {ext}")

    loaded = loader(temp_filepath)
    docs = loaded.load()
    logging.info(docs)
    return docs
