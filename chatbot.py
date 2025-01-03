import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)


class ChatBot:

    def __init__(self):
        self.openai_api_key, self.database_password = self.get_env_info()
        self.chat = ChatOpenAI(model='gpt-3.5-turbo')
        self.messages = []

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        openai_api_key = os.getenv("OPENAI_API_KEY")
        database_password = os.getenv("DATABASE_PASSWORD")
        return openai_api_key, database_password

    def write_prompt(self, prompt):
        new_prompt = HumanMessage(prompt)
        self.messages.append(new_prompt)

    def chat_answer(self):
        res = self.chat.invoke(self.messages)

        return res


