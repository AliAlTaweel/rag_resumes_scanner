"""
Document loading and splitting utilities for resume ingestion.
"""
import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_docs(directory: str) -> List[Document]:
    """
    Load all PDF documents from the given directory.

    Args:
        directory: Path to folder containing PDF resumes.

    Returns:
        List of LangChain Document objects.
    """
    if not os.path.exists(directory):
        logger.info("Directory '%s' not found. Creating it.", directory)
        os.makedirs(directory)

    loader = DirectoryLoader(directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    logger.info("Loaded %d documents from '%s'.", len(documents), directory)
    return documents


def split_docs(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents:    List of LangChain Document objects to split.
        chunk_size:   Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
    return chunks