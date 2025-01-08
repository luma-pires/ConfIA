import streamlit as st
from chatbot_stream import ChatBot


class Application:

    def __init__(self):
        self.chatbot = ChatBot()

    def run(self):
        self.set_interface()
        self.set_history()
        user_input = st.chat_input(placeholder="Me envie uma mensagem para começarmos :)")
        self.interaction(user_input)

    @staticmethod
    def set_interface():
        st.set_page_config(page_title="ConfIA", page_icon="😉")
        st.title("Olá, eu sou a ConfIA 😉")
        st.subheader("Como posso ajudar você hoje?")

    @staticmethod
    def set_history():
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    @staticmethod
    def display_chat():
        for message, response in st.session_state.chat_history:
            st.write("_____________")
            st.write(f"🫵🏽 Você: {message}")
            st.write(f"😉 ConfIA: {response}")

    def interaction(self, user_input):
        if user_input:
            response = self.chatbot.main(user_input)
            st.session_state.chat_history.append((user_input, response))
            self.display_chat()
