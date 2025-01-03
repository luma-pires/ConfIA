import os
from dotenv import load_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_groq import ChatGroq
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from textblob import TextBlob


class ChatBot:

    def __init__(self):
        self.groq_api_key, self.db_api_key = self.get_env_info()
        self.chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
        self.db = pinecone.init(api_key=self.db_api_key, environment="us-west1-gcp")
        self.embeddings_model = OpenAIEmbeddings()
        self.messages = []
        self.index_name = "quickstart"
        self.index = self.get_index()
        self.interaction_id = -1

    @staticmethod
    def get_env_info():
        load_dotenv('./.env')
        groq_api_key = os.getenv("GROQ_API_KEY")
        database_password = os.getenv("DB_API_KEY")
        return groq_api_key, database_password

    def creating_index(self):
        self.db.create_index(
            name=self.index_name,
            dimension=1024,
            metric="cosine"
        )

    def get_index(self):
        return self.db.Index(self.index_name)

    def search_similar_context(self, query_text):
        query_embedding = self.embeddings_model.embed(query_text)
        result = self.index.query(
            queries=[query_embedding.tolist()],
            top_k=3,
            include_values=True
        )
        return result

    @staticmethod
    def get_sentiment(feedback):
        return TextBlob(feedback).sentiment.polarity

    def write_prompt(self, user_prompt):
        new_prompt = HumanMessage(user_prompt)
        self.messages.append(new_prompt)

    def chat_answer(self):
        answer = self.chat.invoke(self.messages)
        return answer

    def interaction_with_chat(self, user_prompt):
        self.write_prompt(user_prompt)
        ai_answer = self.chat_answer()
        print(ai_answer)
        return user_prompt, ai_answer

    def store_interaction_in_db(self, user_prompt, ai_answer):
        user_embedding = self.embeddings_model.embed(user_prompt)
        ai_answer_embedding = self.embeddings_model.embed(ai_answer)
        self.interaction_id = self.interaction_id + 1
        self.index.upsert(
            [(self.interaction_id, user_embedding.tolist()), (self.interaction_id, ai_answer_embedding.tolist())]
        )

    def learn_from_user_feedback(self, user_message, feedback):
        sentiment = self.get_sentiment(feedback)
        if sentiment > 0:
            ai_answer = self.chat_answer()
            self.store_interaction_in_db(user_message, ai_answer)

    def first_touch(self):
        user_prompt = input("Talk to me!")
        self.interaction_with_chat(user_prompt)

    def main(self):
        self.first_touch()
        while True:
            user_prompt = input("Talk to me!")
            self.learn_from_user_feedback(user_prompt)


a = ChatBot()
a.write_prompt("Olá")
a.chat_answer()
