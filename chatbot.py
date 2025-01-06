from db import DataBase
from check_correction import Check_Correction
import os
from dotenv import load_dotenv
from langchain.schema import (HumanMessage)
from langchain_groq import ChatGroq
from embedding import Embedding


class ChatBot(Check_Correction):

    def __init__(self):
        super().__init__()
        self.groq_api_key, self.db_api_key = self.get_env_info()
        self.messages = []
        self.db = DataBase(self.db_api_key)
        self.chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
        self.chat_classifier_continuous_learning = self.chat_classifier_valid_correction = self.chat
        self.prev_question = self.prev_answer = None
        self.db.restarting_indexes()
        self.embeddings_model = Embedding()

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        database_password = os.getenv("DB_API_KEY")
        return groq_api_key, database_password

    def classifier_preference(self, user_message):

        preference_labels = {'Envolve preferências sobre como a resposta deve ser dada': True,
                             'Não envolve preferências sobre como a resposta deve ser dada': False}

        prompt = (
            f"Classifique a seguinte mensagem entre as classes:\n"
            f'{" ".join(preference_labels.keys())}\n'
            f'Mensagem = {user_message} \n'
            f'Retorne apenas o nome da categoria mais provável.\n'
            f'Exemplo de resposta: Categoria X'
        )

        detected_class = self.chat_classifier_continuous_learning.invoke(prompt)
        preference_class = preference_labels.get(detected_class.content, False)

        return preference_class

    def classifier_valid_info(self, user_correction):
        valid_correction = self.validate_user_info(user_correction, self.chat)
        return valid_correction

    def interaction(self, user_prompt):
        self.write_prompt(user_prompt)
        ai_answer = self.informed_chat_answer(user_prompt)
        saving_interaction_in_db = self.messages[-2] + '; ' + self.messages[-1]
        self.db.store_interaction_in_db(saving_interaction_in_db, self.db.index_corrections, 'interaction')
        return ai_answer

    def write_prompt(self, user_prompt):
        self.messages.append(f"Input do usuário: {user_prompt}")
        is_preference = self.classifier_preference(HumanMessage(user_prompt))
        if is_preference:
            self.db.store_interaction_in_db(user_prompt, self.db.index_preferences, 'preference')
        if self.classifier_valid_info(user_prompt):
            self.db.store_interaction_in_db(user_prompt, self.db.index_corrections, 'info')
        else:
            self.messages[-1] = self.messages[-1] + '. A informação desse input é falsa'
            print(self.messages[-1])

    def prompt_for_retrieving_relevant_info(self, user_query):

        context = ''
        query_vector = self.embeddings_model.embeddings_model.encode(user_query)

        # Preferences:
        preferences_db = self.db.index_preferences.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True
        )

        if not ('matches' not in preferences_db or len(preferences_db['matches']) == 0):
            relevant_responses_preferences = [res['metadata']['original_question'] for res in preferences_db['matches']]
            relevant_responses_preferences = list(set(relevant_responses_preferences))
            relevant_responses_preferences = [i for i in relevant_responses_preferences if i != user_query]
            context = context + ' ' + 'Preferências do usuário: ' + '\n'.join(relevant_responses_preferences) \
                if relevant_responses_preferences else context

        # Corrections:
        corrections_db = self.db.index_corrections.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_metadata=True
        )

        if not ('matches' not in corrections_db or len(corrections_db['matches']) == 0):
            relevant_responses_corrections = [res['metadata']['original_question'] for res in corrections_db['matches']]
            relevant_responses_corrections = list(set(relevant_responses_corrections))
            relevant_responses_corrections = [i for i in relevant_responses_corrections if i != user_query]
            context = context + ' ' + 'Mais informações de contexto: ' + '\n'.join(relevant_responses_corrections) \
                if relevant_responses_corrections else context

        prompt = (f"Baseado no seguinte contexto/histórico:\n{context}\n\n"
                  f"Retorne a melhor resposta para o input a seguir (foque nessa pergunta): {user_query}") if context != '' \
            else f"Retorne a melhor resposta para o input a seguir (foque nessa pergunta): {user_query}"
        return prompt

    def informed_chat_answer(self, user_message):
        prompt = self.prompt_for_retrieving_relevant_info(user_message)
        context = self.get_context(prompt)
        ai_answer = self.chat.invoke(context).content
        self.messages.append(f"Output da IA: {ai_answer}")
        return ai_answer

    def get_context(self, prompt, k=4):
        """À medida que o histórico de mensagens cresce, o contexto necessário para gerar respostas pode se tornar
        muito grande, afetando a qualidade das respostas e aumentando o tempo e custo de processamento. Para
        otimizar, o modelo considerará apenas os últimos 2 inputs do usuário e as 2 respostas da IA, totalizando 4
        mensagens. Esse parâmetro (k) pode ser ajustado futuramente.
        """
        messages_considered = self.messages[-k:]
        return f"Tendo em vista o histórico das mensagens, nessa ordem exata: [{";".join(messages_considered)}] {prompt}"

    def main(self, user_prompt):
        ai_answer = self.interaction(user_prompt)
        self.prev_question = user_prompt
        self.prev_answer = ai_answer
        return ai_answer
