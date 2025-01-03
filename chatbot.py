import os
from dotenv import load_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_groq import ChatGroq
from pinecone import Pinecone


class ChatBot:

    def __init__(self):
        self.groq_api_key, self.db_api_key = self.get_env_info()
        self.chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
        self.db = Pinecone(api_key=self.db_api_key)
        self.messages = []

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        database_password = os.getenv("DB_API_KEY")
        return groq_api_key, database_password

    def write_prompt(self, prompt):
        new_prompt = HumanMessage(prompt)
        self.messages.append(new_prompt)

    def chat_answer(self):
        res = self.chat.invoke(self.messages)
        print(res.content)


a = ChatBot()
a.write_prompt("Olá")
a.chat_answer()


