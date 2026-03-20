from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import pandas as pd
import os

print("=== SISTEMA DE IA PARA ANÁLISE DE DADOS ===\n")

# Teste básico de importação
print("✅ Bibliotecas importadas com sucesso!")

# Testar conexão com Ollama
try:
    model = OllamaLLM(model="llama3.2")
    print("✅ Conectado ao Ollama com sucesso!")
except Exception as e:
    print(f"❌ Erro ao conectar ao Ollama: {e}")
    print("Certifique-se de que o Ollama está instalado e rodando")
    exit()

# Criar um exemplo simples com dados fictícios
print("\n--- TESTE COM DADOS SIMPLES ---")

# Criar um DataFrame de exemplo
dados = {
    'produto': ['Notebook', 'Mouse', 'Teclado', 'Monitor', 'Webcam'],
    'vendas': [10, 50, 30, 15, 25],
    'preco': [3500, 80, 150, 1200, 300]
}
df = pd.DataFrame(dados)

print("\n📊 Dados carregados:")
print(df)

# Criar documentos a partir do DataFrame
documents = []
for i, row in df.iterrows():
    content = f"Produto: {row['produto']}, Vendas: {row['vendas']}, Preço: R${row['preco']}"
    doc = Document(page_content=content, metadata={"produto": row['produto']})
    documents.append(doc)

# Criar embeddings e vector store
print("\n🔄 Criando banco vetorial...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_test_db"
)

# Configurar o prompt
template = """
Você é um assistente especializado em análise de dados.
Baseado APENAS nas informações fornecidas, responda a pergunta.

Informações disponíveis:
{context}

Pergunta: {question}

Resposta:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

# Função para fazer perguntas
def perguntar(question):
    print(f"\n❓ Pergunta: {question}")
    
    # Buscar documentos relevantes
    relevant_docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Gerar resposta
    response = chain.invoke({"context": context, "question": question})
    print(f"💡 Resposta: {response}")
    return response

# Testar com algumas perguntas
print("\n=== TESTANDO PERGUNTAS ===")
perguntar("Qual produto vendeu mais?")
perguntar("Qual é o preço do Mouse?")
perguntar("Qual produto tem o maior preço?")

print("\n✅ Sistema configurado com sucesso!")
print("\nAgora você pode substituir os dados de exemplo pelo seu próprio CSV.")
print("Basta modificar o código para carregar seu arquivo com pd.read_csv('seu_arquivo.csv')")