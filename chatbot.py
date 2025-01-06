from embedding import Embedding
from db import DataBase
from check_correction import Check_Correction
import os
from dotenv import load_dotenv
from langchain.schema import (HumanMessage)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq


class ChatBot(Embedding, DataBase, Check_Correction):

    def __init__(self):
        super().__init__()
        self.groq_api_key, self.openai_api_key, self.db_api_key = self.get_env_info()
        self.messages = []
        self.chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
        self.chat_classifier_continuous_learning = self.chat_classifier_valid_correction = self.chat
        self.prev_question = self.prev_answer = None
        self.restarting_indexes()
        self.check_correction_tool = [TavilySearchResults(max_results=3)]

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        database_password = os.getenv("DB_API_KEY")
        return groq_api_key, openai_api_key, database_password

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
            self.store_interaction_in_db(self.embeddings_model, user_prompt, self.index_preferences, 'preference')
        if self.classifier_valid_correction(user_prompt):
            self.store_interaction_in_db(self.embeddings_model, user_prompt, self.index_corrections, 'correction')

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


a = ChatBot()
