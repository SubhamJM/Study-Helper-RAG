import os
from typing import List, Any
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding import get_embedding_manager
from vectorstore import get_vector_store


def load_pdfs(file_path: str = "./data/Theory of Computation.pdf"):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    print(f"{len(docs)} documents loaded")
    return docs


def text_splitter(docs: List[Any], chunk_size: int = 4000, chunk_overlap: int = 500):
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '']
    )
    split_docs = textsplitter.split_documents(docs)
    print(f"{len(docs)} documents split into {len(split_docs)} chunks")
    return split_docs


def run_ingestion(file_path: str = "./data/Theory of Computation.pdf"):
    docs = load_pdfs(file_path)
    chunks = text_splitter(docs)
    texts = [chunk.page_content for chunk in chunks]

    embedding_manager = get_embedding_manager()
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks, embeddings)
    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
