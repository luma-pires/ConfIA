from sentence_transformers import SentenceTransformer


class Embedding:

    def __init__(self):

        self.available_models = {'BERT': 'bert-base-nli-mean-tokens',
                                 'DistilBERT': 'distilbert-base-nli-stsb-mean-tokens',
                                 'RoBERTa': 'roberta-base-nli-stsb-mean-tokens',
                                 'MiniLM': 'all-MiniLM-L6-v2',
                                 'T5': 't5-small'}
        self.model = 'BERT'
        self.embeddings_model = SentenceTransformer(self.available_models[self.model])
