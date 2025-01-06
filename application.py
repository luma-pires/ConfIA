import streamlit as st
from chatbot import ChatBot


class Application:

    def __init__(self):
        self.chatbot = ChatBot()
        self.messages_user = []
        self.messages_ai = []

    def run(self):
        self.set_interface()
        self.set_history()

        user_input = st.chat_input(placeholder="Me envie uma mensagem para começarmos :)")
        self.interaction(user_input)

    @staticmethod
    def set_interface():
        st.set_page_config(page_title="ConfIA", page_icon="😉")
        st.title("Olá, sou a ConfIA 😉")
        st.header("Como posso ajudar você hoje?")

    @staticmethod
    def set_history():
        if 'history' not in st.session_state:
            st.session_state.history = []

    def display_chat(self):
        for i in range(0, len(self.messages_user)):
            message = self.messages_user[i]
            response = self.messages_ai[i]
            st.write(f"🫵🏽 Você: {message}")
            st.write(f"😉 ConfIA: {response}")

    def interaction(self, user_input):
        if user_input:
            response = self.chatbot.main(user_input)
            self.messages_user.append(user_input)
            self.messages_ai.append(response)
            self.display_chat()
