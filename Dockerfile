FROM python:3.12-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos do projeto para o contêiner
COPY . .

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pinecone \
    python-dotenv \
    langchain \
    streamlit \
    sentence-transformers \
    langchain-groq \
    torch

# Exponha a porta que o Streamlit usará
EXPOSE 8501

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "main.py"]
