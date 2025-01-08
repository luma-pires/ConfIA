# Título
ConfIA: um chatbot de Aprendizado Contínuo

![confia](https://github.com/user-attachments/assets/f8063271-e1a8-4b7a-9f78-9bc0f8bb9490)

# Resumo
Este projeto programa o comportamento do ConfIA, um chatbot capaz de aprender com as interações do usuário. Ele armazena as preferências de resposta do usuário, dentre outras informações fornecidas, em um banco de dados vetorial (Pinecone). O Chatbot é programado para não aceitar informações falaciosas. Essa verificação é feita a partir da classificação da informação fornecida entre falaciosa ou não falaciosa. A interface visual do Chatbot foi desenvolvida utilizando o Streamlit e o LLM utilizado é o Llama.

# Modelo de Grafos
![image](https://github.com/user-attachments/assets/28171b40-9697-49a1-8bcd-f1c8921183d5)


# Primeiros Passos
Certifique-se de adicionar um arquivo .env no diretório do projeto, contendo as seguintes informações:

```python
GROQ_API_KEY=your_groq_api_key
DB_API_KEY=your_db_api_key
```

# Rodando Localmente
O primeiro passo, claro, é instalar as dependências necessárias:
```python
pip install -r requirements.txt
```

Após isso, rode no terminal:
```python
streamlit run ./caminho/para/o/projeto/main.py
```
A interface visual do ConfIA será aberta automaticamente em seu navegador.

# Rodando com o Docker
Certifique-se de que tem o Docker Desktop Instalado. Após isso, criar uma imagem com base no Dockerfile do projeto:
```python
docker build -t nome_da_imagem .
```

Após isso, rodar:
```python
docker run --name nome_do_container -p 8501:8501 nome_da_imagem
```
Com isso, clicar em um dos links (Local ou Network URL) que irão aparecer no terminal. Ele levará à interface gráfica do ConfIA no seu navegador.

# Informações adicionais

LLM: Llama 3.1

Base de Dados Vetorial: Pinecone

Encoder: BERT
