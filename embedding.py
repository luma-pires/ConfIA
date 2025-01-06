from sentence_transformers import SentenceTransformer


class Embedding:

    def __init__(self):
        self.embeddings_model = SentenceTransformer('bert-base-nli-mean-tokens')
