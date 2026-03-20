import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os

print("🔍 SISTEMA DE ANÁLISE DE DADOS COM IA")
print("-" * 50)

# ============================================
# CONFIGURAÇÕES - VOCÊ SÓ PRECISA MUDAR AQUI
# ============================================

# 1. Nome do seu arquivo (coloque na pasta 'dados')
ARQUIVO_DADOS = "dados/seu_arquivo.csv"  # Mude para seu arquivo!

# 2. Colunas do seu CSV (ajuste conforme necessário)
COLUNA_TEXTO = "descricao"  # Coluna com o texto principal
COLUNAS_METADADOS = ["categoria", "data"]  # Colunas para filtrar

# 3. Modelo a ser usado
MODELO_LLM = "llama3.2"  # ou "deepseek-r1", "mistral", etc.
MODELO_EMBEDDING = "mxbai-embed-large"

# ============================================
# NÃO PRECISA MUDAR NADA ABAIXO (a menos que queira)
# ============================================

print(f"\n📂 Carregando dados: {ARQUIVO_DADOS}")

try:
    df = pd.read_csv(ARQUIVO_DADOS)
    print(f"✅ {len(df)} linhas carregadas")
    print(f"📊 Colunas disponíveis: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {ARQUIVO_DADOS}")
    print("\nCertifique-se de que:")
    print(f"1. O arquivo existe na pasta 'dados/'")
    print("2. O nome do arquivo está correto")
    print("3. O arquivo está no formato CSV")
    exit()

# Conectar ao Ollama
print("\n🔄 Conectando ao Ollama...")
try:
    llm = OllamaLLM(model=MODELO_LLM)
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    print(f"✅ Modelos carregados: {MODELO_LLM} e {MODELO_EMBEDDING}")
except Exception as e:
    print(f"❌ Erro: {e}")
    print("Verifique se o Ollama está rodando (ícone na bandeja do sistema)")
    exit()

# Criar banco vetorial
print("\n🔄 Criando banco de conhecimento...")
documents = []
for idx, row in df.iterrows():
    # Criar texto combinando todas as colunas relevantes
    texto = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    
    # Criar metadados para filtros
    metadados = {}
    for col in COLUNAS_METADADOS:
        if col in df.columns:
            metadados[col] = str(row[col])
    
    documents.append(Document(page_content=texto, metadata=metadados))

# Criar ou carregar banco vetorial
db_path = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=db_path
)
print(f"✅ Banco criado com {len(documents)} documentos")

# Configurar prompt
template = """
Você é um analista de dados especialista.
Use APENAS as informações fornecidas abaixo para responder.

INFORMAÇÕES RELEVANTES:
{context}

PERGUNTA: {question}

Responda de forma clara e objetiva, baseando-se apenas nos dados fornecidos.
Se a informação não estiver disponível, diga que não encontrou nos dados.

RESPOSTA:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

# Loop interativo
print("\n" + "="*50)
print("💬 Sistema pronto! Faça suas perguntas")
print("Digite 'sair' para encerrar")
print("Dica: pergunte sobre tendências, comparações, totais, etc.")
print("="*50)

while True:
    pergunta = input("\n📝 Sua pergunta: ").strip()
    
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("👋 Até mais!")
        break
    
    if not pergunta:
        continue
    
    print("🔍 Analisando...")
    
    # Buscar documentos relevantes
    docs = vector_store.similarity_search(pergunta, k=5)
    contexto = "\n---\n".join([doc.page_content for doc in docs])
    
    # Gerar resposta
    resposta = chain.invoke({"context": contexto, "question": pergunta})
    
    print(f"\n💡 Resposta: {resposta}")
    print("-" * 50)