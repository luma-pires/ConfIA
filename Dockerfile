FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pinecone \
    python-dotenv \
    langchain \
    streamlit \
    sentence-transformers \
    langchain-groq \
    torch
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
