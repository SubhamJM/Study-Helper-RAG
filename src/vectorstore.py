import chromadb
import os
import uuid
from typing import List, Any
import numpy as np


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./vector_store", description: str = "This vectorDB stores all the information regarding the subject Theory of Computation"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.description = description
        self._collection = None
        self._client = None

    def _initiate_db(self):
        if self._client is not None:
            return
        try:
            print(f"Initiating vectordb with name {self.collection_name}")
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": self.description}
            )
            print(f"Vectordb ready at {self.persist_directory}")
        except:
            print("Error creating vectordb!")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        self._initiate_db()
        if len(documents) != len(embeddings):
            raise ValueError("Length of documents and embeddings must be same")

        print(f"Adding {len(documents)} documents to the vectordb")

        ids_list = []
        metadatas_list = []
        documents_list = []
        embeddings_list = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids_list.append(f"{uuid.uuid4().hex[:8]}_{i}")
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i + 1
            metadata["doc_length"] = len(doc.page_content)
            metadatas_list.append(metadata)
            documents_list.append(doc.page_content)
            embeddings_list.append(emb.tolist())

        try:
            self._collection.add(
                ids=ids_list,
                documents=documents_list,
                embeddings=embeddings_list,
                metadatas=metadatas_list
            )
            print("Added to vectordb successfully!")
        except:
            print("Error adding to vectordb!")
            raise

    def query(self, query_embeddings, n_results: int = 5):
        self._initiate_db()
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )


_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
