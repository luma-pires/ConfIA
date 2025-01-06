from pinecone import Pinecone, ServerlessSpec
from datetime import datetime


class DataBase:

    def __init__(self, db_api_key):
        self.db_api_key = db_api_key
        self.db = Pinecone(api_key=self.db_api_key, environment="us-west1-gcp")
        self.checking_indexes()
        self.index_preferences = self.get_index('index-preferences')
        self.index_corrections = self.get_index('index-corrections')

    def checking_indexes(self):
        """ Cria os indexes dn banco de dados vetorial, caso já não existam. Isso evita erros caso o index não exista"""
        self.creating_index_if_it_does_not_exists('index-preferences')
        self.creating_index_if_it_does_not_exists('index-corrections')

    def restarting_indexes(self):
        """ Quando o chat se reinicia, é preciso apagar a memória armazenada da interação anterior. Caso contrário ela
        pode interferir nas futuras respostas do chatbot"""
        self.erase_index_content(self.index_preferences)
        self.erase_index_content(self.index_corrections)

    def creating_index_if_it_does_not_exists(self, index_name):
        existing_indexes = [i['name'] for i in self.db.list_indexes().to_dict()['indexes']]
        if index_name not in existing_indexes:
            self.db.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    def get_index(self, index_name):
        return self.db.Index(index_name)

    @staticmethod
    def erase_index_content(index):
        """ Apaga o conteúdo do index, mas não o index em si, pois será reaproveitado em usos futuros do chatbot"""
        stats_index = index.describe_index_stats()
        n_vectors = stats_index.get("total_vector_count", 0)
        index.delete(delete_all=True) if n_vectors != 0 else None

    @staticmethod
    def store_interaction_in_db(embeddings_model, user_prompt, index, interaction):
        """ Armazena informações no banco de dados vetorial"""
        user_embedding = embeddings_model.encode(user_prompt)
        data_id = str(interaction) + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        index.upsert(
            [(data_id, user_embedding.tolist(), {'original_question': user_prompt})]
        )
