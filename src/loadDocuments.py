import os
from typing import List, Any
from langchain_community.document_loaders import PDFPlumberLoader
def load_pdfs(file_path: str = "./data/Theory of Computation.pdf"):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    print(f"{len(docs)} documents have been loaded")
    return docs

from langchain_text_splitters import RecursiveCharacterTextSplitter
def text_splitter(docs: List[Any], chunk_size: int = 4000, chunk_overlap: int = 500):
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '']
    )

    split_docs = textsplitter.split_documents(docs)
    print(f"{len(docs)} documents have been split into {len(split_docs)} chunks")

    return split_docs

docs = load_pdfs()
chunks = text_splitter(docs)
texts = [chunk.page_content for chunk in chunks]

from embedding import embedding_manager
embeddings = embedding_manager.generate_embeddings(texts)

from vectorstore import vector_store
vector_store.add_documents(chunks, embeddings)