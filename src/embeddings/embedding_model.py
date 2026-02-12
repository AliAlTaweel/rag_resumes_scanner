"""
Embedding model configuration and factory.
"""
import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model.

    Args:
        model_name: HuggingFace model identifier for sentence embeddings.

    Returns:
        HuggingFaceEmbeddings instance ready for use.
    """
    logger.info("Loading embedding model: %s", model_name)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    logger.info("Embedding model loaded successfully.")
    return embeddings