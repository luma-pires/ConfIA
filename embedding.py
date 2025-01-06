from sentence_transformers import SentenceTransformer


class Embedding:

    def __init__(self):
        self.embeddings_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def search_similar_context(self, query_text, index, k=3):
        query_embedding = self.embeddings_model.embed(query_text)
        result = index.query(
            queries=[query_embedding.tolist()],
            top_k=k,
            include_values=True
        )
        return result
