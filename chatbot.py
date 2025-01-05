import os
from dotenv import load_dotenv
from langchain.schema import (HumanMessage)
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datetime import datetime


class ChatBot:

    def __init__(self):
        self.groq_api_key, self.openai_api_key, self.db_api_key = self.get_env_info()
        self.chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
        self.chat_classifier_continuous_learning = self.chat_classifier_valid_correction = self.chat
        self.db = Pinecone(api_key=self.db_api_key, environment="us-west1-gcp")
        self.embeddings_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.messages = []
        self.checking_indexes()
        self.index_preferences = self.get_index('index-preferences')
        self.index_corrections = self.get_index('index-corrections')
        self.prev_question = self.prev_answer = None
        self.restarting_indexes()

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        database_password = os.getenv("DB_API_KEY")
        return groq_api_key, openai_api_key, database_password

    def checking_indexes(self):
        self.creating_index_if_it_does_not_exists('index-preferences')
        self.creating_index_if_it_does_not_exists('index-corrections')

    def restarting_indexes(self):
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

    def search_similar_context(self, query_text, index, k=3):
        query_embedding = self.embeddings_model.embed(query_text)
        result = index.query(
            queries=[query_embedding.tolist()],
            top_k=k,
            include_values=True
        )
        return result

    def classifier_preference(self, user_message):

        preference_labels = {'Envolve preferências sobre como a resposta deve ser dada': True,
                             'Não envolve preferências sobre como a resposta deve ser dada': False}

        prompt = f"Classifique a seguinte mensagem entre as classes\
               {" ".join(preference_labels.keys())}\
               Mensagem = {user_message} \n\
               Retorne apenas o nome da categoria mais provável.\
               Exemplo de resposta: Categoria X"

        detected_class = self.chat_classifier_continuous_learning.invoke(prompt)
        preference_class = preference_labels.get(detected_class.content, False)

        return preference_class

    def classifier_valid_correction(self, user_correction):
        valid_correction = False
        if self.prev_answer is not None and self.prev_question is not None:
            prompt = (f"Considerando o diálogo abaixo, a correção do usuário está correta? Responda com Sim ou Não: \n"
                      f"Usuário: {self.prev_question} \n"
                      f"Resposta: {self.prev_answer}' \n"
                      f"Usuário: {user_correction}")
            corrections_labels = {'sim': True, 'não': False}
            detected_classes = self.chat_classifier_valid_correction.invoke(prompt)
            answer = detected_classes.content.replace('.', '').lower()
            valid_correction = corrections_labels.get(answer, False)
        return valid_correction

    def interaction(self, user_prompt):
        self.write_prompt(user_prompt)
        ai_answer = self.informed_chat_answer(user_prompt)
        print(ai_answer)
        return ai_answer

    def write_prompt(self, user_prompt):
        self.messages.append(user_prompt)
        is_preference = self.classifier_preference(HumanMessage(user_prompt))
        if is_preference:
            self.store_interaction_in_db(user_prompt, self.index_preferences, 'preference')
        if self.classifier_valid_correction(user_prompt):
            self.store_interaction_in_db(user_prompt, self.index_corrections, 'correction')

    def store_interaction_in_db(self, user_prompt, index, interaction):
        user_embedding = self.embeddings_model.encode(user_prompt)
        data_id = str(interaction) + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        index.upsert(
            [(data_id, user_embedding.tolist(), {'original_question': user_prompt})]
        )

    def prompt_for_retrieving_relevant_info(self, user_query):

        context = ''
        query_vector = self.embeddings_model.encode(user_query)

        # Preferences:
        preferences_db = self.index_preferences.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_metadata=True
        )

        if not ('matches' not in preferences_db or len(preferences_db['matches']) == 0):
            relevant_responses_preferences = [res['metadata']['original_question'] for res in preferences_db['matches']]
            relevant_responses_preferences = list(set(relevant_responses_preferences))
            relevant_responses_preferences = [i for i in relevant_responses_preferences if i != user_query]
            context = context + ' ' + 'Preferências do usuário: ' + '\n'.join(relevant_responses_preferences) \
                if relevant_responses_preferences else context

        # Corrections:
        corrections_db = self.index_corrections.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_metadata=True
        )

        if not ('matches' not in corrections_db or len(corrections_db['matches']) == 0):
            relevant_responses_corrections = [res['metadata']['original_question'] for res in corrections_db['matches']]
            relevant_responses_corrections = list(set(relevant_responses_corrections))
            relevant_responses_corrections = [i for i in relevant_responses_corrections if i != user_query]
            context = context + ' ' + 'Correções do usuário: ' + '\n'.join(relevant_responses_corrections) \
                if relevant_responses_corrections else context

        prompt = f"Baseado no seguinte contexto:\n{context}\n\nPergunta: {user_query}" if context != '' else user_query
        self.messages.append(prompt)
        self.messages = list(set(self.messages))

    def informed_chat_answer(self, user_message):
        self.prompt_for_retrieving_relevant_info(user_message)
        ai_answer = self.chat.invoke(self.messages).content
        self.messages.append(ai_answer)
        return ai_answer

    def main(self, user_prompt):
        ai_answer = self.interaction(user_prompt)
        self.prev_question = user_prompt
        self.prev_answer = ai_answer

    @staticmethod
    def erase_index_content(index):
        stats_index = index.describe_index_stats().get("namespaces", {})
        namespaces = stats_index.get("namespaces", {})
        n_vectors = sum(ns.get("vector_count", 0) for ns in namespaces.values())
        index.delete(delete_all=True) if n_vectors != 0 else None


a = ChatBot()
a.main()
