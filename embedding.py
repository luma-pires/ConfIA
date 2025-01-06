from sentence_transformers import SentenceTransformer


class Embedding:

    def __init__(self):

        self.available_models = {'BERT': 'bert-base-nli-mean-tokens'}
        self.model = 'BERT'
        self.embeddings_model = SentenceTransformer(self.available_models[self.model])
