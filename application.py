import streamlit as st
from chatbot import ChatBot


class Application:

    def __init__(self):
        self.chatbot = ChatBot()

    def run(self):

        st.title("Chatbot com Streamlit")
        user_input = st.text_input("Digite sua pergunta:")

        if user_input:
            response = self.chatbot.main(user_input)
            st.write("Chatbot: ", response)

        st.write(f"Última pergunta: {self.chatbot.prev_question}")
        st.write(f"Última resposta: {self.chatbot.prev_answer}")