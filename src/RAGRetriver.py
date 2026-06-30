from src.vectorstore import VectorStore, get_vector_store
from src.embedding import EmbeddingManager, get_embedding_manager


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrive_context(self, query: str, top_k: int = 5):
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k
            )

            context = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                ids = results["ids"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for i, (document, ID, metadata, distance) in enumerate(zip(documents, ids, metadatas, distances)):
                    context.append({
                        "id": ID,
                        "content": document,
                        "metadata": metadata,
                        "distance": distance,
                        "rank": i + 1
                    })
            else:
                print("No documents found")

            return context
        except:
            print("Error retrieving data")
            raise


_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_vector_store(), get_embedding_manager())
    return _retriever
