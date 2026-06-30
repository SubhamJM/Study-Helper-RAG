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
        self.collection = None
        self.client = None
        self._initiate_db()

    def _initiate_db(self):
        try:
            print(f"Initiating vectordb with name {self.collection_name} and description: {self.description}")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":self.description}
            )

            print(f"vectordb created successfully at {self.persist_directory}")
        except:
            print("error creating vectordb")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Length of documents and embeddings must be same")
        
        print(f"Addings {len(documents)} documents to the vectordb")

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
            self.collection.add(
                ids=ids_list,
                documents=documents_list,
                embeddings=embeddings_list,
                metadatas=metadatas_list
            )
            print("Added to vectordb successfully!")
        except:
            print("Error adding to vectordb!")
            raise

vector_store = VectorStore()