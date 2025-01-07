import os
from dotenv import load_dotenv
from db import DataBase
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq


class State(TypedDict):
    messages: Annotated[list, add_messages]


class Graph(DataBase):

    def __init__(self):

        super().__init__()
        self.groq_api_key = self.get_llm_env_info()
        self.llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")

        self.latest_user_message = None
        self.index_insert_db = None
        self.context = None
        self.messages = []

        self.graph_builder = StateGraph(State)
        self.built_graph()
        self.graph = self.graph_builder.compile()

    def main(self, user_prompt):
        self.latest_user_message = user_prompt
        self.messages.append(self.latest_user_message)
        state = {
            "messages": [{"role": "user", "content": self.latest_user_message}]
        }
        self.graph.run(state)
        ai_answer = self.messages[-1]
        return ai_answer

    @staticmethod
    def get_llm_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        return groq_api_key

    def built_graph(self):

        # Nodes/Nós:

        self.graph_builder.add_node("classify_preference", self.classifying_preference)
        self.graph_builder.add_node("validate_info", self.classifying_valid_info)
        self.graph_builder.add_node("save_in_db", self.saving_in_db)
        self.graph_builder.add_node("context", self.get_context)
        self.graph_builder.add_node("answer", self.answer)
        self.graph_builder.add_node("answer_info_incorrect", self.deal_with_incorrect_input)
        self.graph_builder.add_node("answer_preference", self.answer_preference)

        # Edges/Arestas:

        self.graph_builder.add_edge(START, "classify_preference")

        self.graph_builder.add_conditional_edge(
            "classify_preference",
            "answer_preference",
            condition=lambda state: state.get("classify_preference", True)
        )

        self.graph_builder.add_edge("answer_preference", "save_in_db")

        self.graph_builder.add_edge("save_in_db", "context")

        self.graph_builder.add_edge("context", "answer")

        self.graph_builder.add_conditional_edge(
            "classify_preference",
            "validate_info",
            condition=lambda state: state.get("classify_preference", False)
        )

        self.graph_builder.add_conditional_edge(
            "validate_info",
            "save_in_db",
            condition=lambda state: state.get("validate_info", True)
        )

        self.graph_builder.add_conditional_edge(
            "validate_info",
            "answer_info_incorrect",
            condition=lambda state: state.get("validate_info", False)
        )

        self.graph_builder.add_edge("answer", END)

        self.graph_builder.add_edge("answer_info_incorrect", END)

    def classifying_preference(self, state: State):

        preference_labels = {'Envolve preferências sobre como a resposta deve ser dada': True,
                             'Não envolve preferências sobre como a resposta deve ser dada': False}

        prompt = (
            f"Classifique a seguinte mensagem entre as classes:\n"
            f'{" ".join(preference_labels.keys())}\n'
            f'Mensagem = {self.latest_user_message} \n'
            f'Retorne apenas a categoria mais provável'
        )

        detected_class = self.llm.invoke(prompt).content
        answer = detected_class.replace('.', '').lower()

        is_preference = preference_labels.get(answer, False)
        self.index_insert_db = self.db.index_preferences if is_preference else self.index_insert_db

        return is_preference

    def classifying_valid_info(self, state: State):

        corrections_labels = {'não': True, 'sim': False}

        prompt = (f"É provável que a mensagem abaixo seja falsa? Responda apenas com sim ou não. Se for algo sobre o"
                  f"usuário, algo subjetivo, retorne não. Mensagem = {self.latest_user_message}")

        detected_class = self.llm.invoke(prompt).content
        answer = detected_class.replace('.', '').lower()

        print(self.latest_user_message)
        print(answer)

        is_valid_correction = corrections_labels.get(answer, False)
        self.index_insert_db = self.db.index_corrections if is_valid_correction else self.index_insert_db

        return is_valid_correction

    def saving_in_db(self, state: State):
        self.db.store_interaction_in_db(self.latest_user_message, self.index_insert_db)

    def get_context(self, state: State):

        related_info_in_db = self.retrieving_related_info_from_past_messages_with_db()
        message_history = self.getting_context_from_latest_k_messages()

        self.context = related_info_in_db + '; ' + message_history

    def retrieving_related_info_from_past_messages_with_db(self):

        context = ''
        user_query = self.latest_user_message

        query_vector = self.embeddings_model.embeddings_model.encode(user_query)

        preferences_db = self.db.index_preferences.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True
        )

        if not ('matches' not in preferences_db or len(preferences_db['matches']) == 0):
            relevant_responses_preferences = [res['metadata']['original_question'] for res in
                                              preferences_db['matches']]
            relevant_responses_preferences = list(set(relevant_responses_preferences))
            relevant_responses_preferences = [i for i in relevant_responses_preferences if i != user_query]
            context = 'Preferências de estilos de resposta do usuário: ' + '\n'.join(relevant_responses_preferences) \
                if relevant_responses_preferences else context

        corrections_db = self.db.index_corrections.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_metadata=True
        )

        if not ('matches' not in corrections_db or len(corrections_db['matches']) == 0):
            relevant_responses_corrections = [res['metadata']['original_question'] for res in
                                              corrections_db['matches']]
            relevant_responses_corrections = list(set(relevant_responses_corrections))
            relevant_responses_corrections = [i for i in relevant_responses_corrections if i != user_query]
            context = (context + ' ' + 'Informações obtidas de mensagens anteriores: ' +
                       '\n'.join(relevant_responses_corrections)) if relevant_responses_corrections else context

        prompt = (f"Baseado no seguinte contexto:\n{context}\n"
                  f"Retorne a melhor resposta para pergunta a seguir (foque nessa pergunta): {user_query}") if context != '' \
            else f"Retorne a melhor resposta para a pergunta a seguir (foque nessa pergunta): {user_query}"

        return prompt

    def getting_context_from_latest_k_messages(self, k=6):
        """À medida que o histórico de mensagens cresce, o contexto necessário para gerar respostas pode se tornar
        muito grande, afetando a qualidade das respostas e aumentando o tempo e custo de processamento. Para
        otimizar, o modelo considerará apenas os últimos 3 inputs do usuário e as 3 respostas da IA, totalizando 6
        mensagens. Esse parâmetro (k) pode ser ajustado futuramente.
        """
        messages_considered = self.messages[-k:]
        return (f"Tendo em vista o histórico das {k} mensagens anteriores, nessa ordem exata: "
                f"[{";\n ".join(messages_considered)}]")

    def response_ai(self):
        response = self.llm.invoke(self.context).content
        self.messages.append(f"Resposta da IA: {response}")

    def answer(self, state: State):
        self.response_ai()

    def deal_with_incorrect_input(self, state: State):
        self.latest_user_message = ('Há grandes chances de que essa informação fornecida pelo usuário seja falsa' + ' '
                                    + self.latest_user_message)
        self.context = self.latest_user_message
        self.response_ai()

    def answer_preference(self, state: State):
        self.context = f'O usuário enviou uma preferência de estilos de resposta: {self.latest_user_message}'
        self.response_ai()
