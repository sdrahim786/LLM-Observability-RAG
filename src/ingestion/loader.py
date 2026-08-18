"""
Ingestion module -- Phase 1 of the Personal RAG Assistant pipeline.

Responsible for turning raw files (PDFs, text notes, Word docs, markdown)
into LangChain `Document` objects: { page_content: str, metadata: dict }.

This module does NOT chunk, embed, or store anything -- it only loads.
Keeping ingestion separate from chunking (Phase 2) means you can test each
step on its own, and swap a loader later without touching the rest of the
pipeline.
"""

from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

# Map file extensions to the LangChain loader class that knows how to read them.
# Add new file types here as you support them -- nothing else needs to change.
#
# Note: .md is loaded with plain TextLoader rather than UnstructuredMarkdownLoader.
# Unstructured's markdown loader pulls in NLTK sentence-tokenizer models it
# downloads on first use -- unnecessary weight for a personal notes folder,
# and it breaks offline/behind a firewall. Chunking (Phase 2) doesn't care
# about markdown structure, so plain text is fine here.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())


def load_file(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a single file into a list of LangChain Documents.

    Note: a PDF loader returns one Document per page; a text/markdown loader
    returns one Document for the whole file. That's expected -- chunking
    (Phase 2) is what normalizes everything into uniform-sized pieces later.
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{extension}' for {file_path.name}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    loader_cls = LOADER_MAP[extension]
    loader = loader_cls(str(file_path))
    documents = loader.load()

    # Tag every document with where it came from and what kind of file it was.
    # This metadata is what lets the assistant cite sources later at retrieval time.
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_type"] = extension.lstrip(".")

    return documents


def load_directory(dir_path: Union[str, Path], recursive: bool = True) -> List[Document]:
    """
    Load every supported file in a directory into one flat list of Documents.
    Unsupported files are skipped (and reported) rather than raising, since a
    real notes folder will always contain files you don't want ingested.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    pattern = "**/*" if recursive else "*"
    all_documents: List[Document] = []
    skipped: List[str] = []

    for file_path in sorted(dir_path.glob(pattern)):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(file_path.name)
            continue
        all_documents.extend(load_file(file_path))

    if skipped:
        print(f"Skipped {len(skipped)} unsupported file(s): {skipped}")

    return all_documents
