import streamlit as st
from chatbot import ChatBot


class Application:

    def __init__(self):
        self.chatbot = ChatBot()

    def run(self):

        st.title("Chatbot com Streamlit")
        st.set_page_config(page_title="Chatbot", page_icon="💬")
        st.header('Basic Chatbot')
        st.write('Allows users to interact with the LLM')
        user_input = st.chat_input(placeholder="Ask me anything!")

        if user_input:
            st.write(user_input, 'user')
            response = self.chatbot.main(user_input)
            st.write(response, 'ai')
