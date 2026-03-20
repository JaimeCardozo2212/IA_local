import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
import json
from sqlalchemy import create_engine, text
import tempfile
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Analista IA - Multi-formatos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILO CSS PERSONALIZADO
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1ECF1;
        color: #0C5460;
        border: 1px solid #BEE5EB;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3CD;
        color: #856404;
        border: 1px solid #FFEEBA;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================

@st.cache_data
def load_csv(file):
    """Carrega arquivo CSV"""
    try:
        df = pd.read_csv(file)
        return df, "CSV carregado com sucesso!"
    except Exception as e:
        return None, f"Erro ao carregar CSV: {str(e)}"

@st.cache_data
def load_excel(file):
    """Carrega arquivo Excel"""
    try:
        df = pd.read_excel(file)
        return df, "Excel carregado com sucesso!"
    except Exception as e:
        return None, f"Erro ao carregar Excel: {str(e)}"

@st.cache_data
def load_json(file):
    """Carrega arquivo JSON"""
    try:
        data = json.load(file)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()
        return df, "JSON carregado com sucesso!"
    except Exception as e:
        return None, f"Erro ao carregar JSON: {str(e)}"

@st.cache_data
def load_sql(connection_string, query):
    """Carrega dados de banco SQL"""
    try:
        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return df, "Dados SQL carregados com sucesso!"
    except Exception as e:
        return None, f"Erro ao carregar SQL: {str(e)}"

@st.cache_data
def load_txt(file):
    """Carrega arquivo TXT como tabela"""
    try:
        # Tenta detectar o separador
        content = file.read().decode('utf-8').split('\n')
        if '\t' in content[0]:
            df = pd.read_csv(file, sep='\t')
        elif ';' in content[0]:
            df = pd.read_csv(file, sep=';')
        elif ',' in content[0]:
            df = pd.read_csv(file, sep=',')
        else:
            # Se não detectar separador, cria uma coluna única
            df = pd.DataFrame({'texto': content})
        return df, "TXT carregado com sucesso!"
    except Exception as e:
        return None, f"Erro ao carregar TXT: {str(e)}"

def create_vector_store(df, embeddings):
    """Cria banco vetorial a partir do DataFrame"""
    documents = []
    for idx, row in df.iterrows():
        # Criar texto com todas as colunas
        texto = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        
        # Criar metadados básicos
        metadados = {
            "indice": idx,
            "total_colunas": len(df.columns)
        }
        
        documents.append(Document(page_content=texto, metadata=metadados))
    
    # Criar banco vetorial
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db_app"
    )
    
    return vector_store, documents

def ask_question(question, vector_store, llm, chain):
    """Faz pergunta ao modelo"""
    # Buscar documentos relevantes
    docs = vector_store.similarity_search(question, k=5)
    contexto = "\n---\n".join([doc.page_content for doc in docs])
    
    # Gerar resposta
    resposta = chain.invoke({"context": contexto, "question": question})
    
    return resposta, docs

# ============================================
# INTERFACE PRINCIPAL
# ============================================

st.markdown('<p class="main-header">📊 Assistente IA Multi-Formatos</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Carregue dados de qualquer fonte e converse com eles</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - CONFIGURAÇÕES
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.title("⚙️ Configurações")
    
    # Seleção do modelo
    modelo_llm = st.selectbox(
        "🤖 Modelo de Linguagem",
        ["llama3.2", "deepseek-r1", "mistral", "phi3"],
        index=0
    )
    
    modelo_embedding = st.selectbox(
        "🔤 Modelo de Embedding",
        ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
        index=0
    )
    
    # Conexão com Ollama
    if st.button("🔄 Conectar ao Ollama", use_container_width=True):
        with st.spinner("Conectando..."):
            try:
                st.session_state.llm = OllamaLLM(model=modelo_llm)
                st.session_state.embeddings = OllamaEmbeddings(model=modelo_embedding)
                st.session_state.ollama_connected = True
                st.success(f"✅ Conectado: {modelo_llm}")
            except Exception as e:
                st.session_state.ollama_connected = False
                st.error(f"❌ Erro: {str(e)}")
    
    # Status da conexão
    if st.session_state.get('ollama_connected', False):
        st.markdown('<div class="success-box">✅ Ollama conectado</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Ollama não conectado</div>', unsafe_allow_html=True)
        st.info("💡 Instale o Ollama em: https://ollama.com")
    
    st.divider()
    
    # Sobre o app
    st.markdown("### 📌 Sobre")
    st.markdown("""
    Este app permite:
    - 📂 Múltiplos formatos (CSV, Excel, JSON, TXT, SQL)
    - 🤖 IA local com Ollama
    - 📊 Visualizações automáticas
    - 💬 Conversa com seus dados
    - 📈 Estatísticas descritivas
    """)

# ============================================
# ÁREA PRINCIPAL - UPLOAD DE DADOS
# ============================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload Arquivo", "🗄️ Banco SQL", "🔗 URL", "📋 Exemplo", "ℹ️ Ajuda"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Escolha um arquivo",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help="Formatos suportados: CSV, Excel, JSON, TXT"
        )
    
    with col2:
        st.markdown("### 🔧 Opções")
        if uploaded_file:
            file_details = {
                "Nome": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamanho": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
    
    if uploaded_file:
        # Determinar tipo do arquivo
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Carregando arquivo..."):
            if file_ext == 'csv':
                df, msg = load_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df, msg = load_excel(uploaded_file)
            elif file_ext == 'json':
                df, msg = load_json(uploaded_file)
            elif file_ext == 'txt':
                df, msg = load_txt(uploaded_file)
            else:
                df, msg = None, "Formato não suportado"
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.markdown(f'<div class="success-box">✅ {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">⚠️ {msg}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔌 Conexão com Banco de Dados")
    
    db_type = st.selectbox(
        "Tipo de Banco",
        ["SQLite", "MySQL", "PostgreSQL", "SQL Server"]
    )
    
    if db_type == "SQLite":
        db_file = st.file_uploader("Arquivo SQLite", type=['db', 'sqlite', 'sqlite3'])
        if db_file:
            # Salvar temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                tmp.write(db_file.getvalue())
                connection_string = f"sqlite:///{tmp.name}"
    
    elif db_type == "MySQL":
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", "localhost")
            user = st.text_input("Usuário")
            database = st.text_input("Database")
        with col2:
            port = st.text_input("Porta", "3306")
            password = st.text_input("Senha", type="password")
        
        if all([host, user, database]):
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    
    elif db_type == "PostgreSQL":
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", "localhost")
            user = st.text_input("Usuário")
            database = st.text_input("Database")
        with col2:
            port = st.text_input("Porta", "5432")
            password = st.text_input("Senha", type="password")
        
        if all([host, user, database]):
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    query = st.text_area("SQL Query", "SELECT * FROM tabela LIMIT 100", height=100)
    
    if st.button("📥 Carregar Dados SQL", use_container_width=True):
        if 'connection_string' in locals():
            with st.spinner("Executando query..."):
                df, msg = load_sql(connection_string, query)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.markdown(f'<div class="success-box">✅ {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warning-box">⚠️ {msg}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 🌐 Carregar de URL")
    url = st.text_input("URL do arquivo (CSV, JSON, Excel)")
    
    if url and st.button("📥 Baixar e Carregar"):
        with st.spinner("Baixando arquivo..."):
            try:
                if url.endswith('.csv'):
                    df = pd.read_csv(url)
                elif url.endswith('.json'):
                    df = pd.read_json(url)
                elif url.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(url)
                else:
                    st.error("Formato não suportado. Use CSV, JSON ou Excel.")
                    st.stop()
                
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Dados carregados da URL!")
            except Exception as e:
                st.error(f"Erro ao carregar: {str(e)}")

with tab4:
    st.markdown("### 📋 Dados de Exemplo")
    
    example = st.selectbox(
        "Escolha um dataset de exemplo",
        ["Vendas", "Clientes", "Produtos", "Financeiro"]
    )
    
    if example == "Vendas":
        data = {
            'data': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'produto': ['Notebook', 'Mouse', 'Teclado', 'Monitor', 'Webcam'] * 20,
            'vendas': np.random.randint(1, 50, 100),
            'preco': np.random.uniform(50, 5000, 100).round(2),
            'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], 100)
        }
    elif example == "Clientes":
        data = {
            'id_cliente': range(1, 101),
            'idade': np.random.randint(18, 70, 100),
            'cidade': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'BA'], 100),
            'renda': np.random.uniform(2000, 15000, 100).round(2),
            'compras_2024': np.random.randint(0, 20, 100)
        }
    elif example == "Produtos":
        data = {
            'codigo': [f'P{str(i).zfill(3)}' for i in range(1, 51)],
            'nome': [f'Produto {i}' for i in range(1, 51)],
            'categoria': np.random.choice(['Eletrônicos', 'Vestuário', 'Alimentos', 'Livros'], 50),
            'estoque': np.random.randint(0, 500, 50),
            'preco_custo': np.random.uniform(10, 1000, 50).round(2),
            'preco_venda': np.random.uniform(20, 2000, 50).round(2)
        }
    elif example == "Financeiro":
        data = {
            'data': pd.date_range(start='2024-01-01', periods=90, freq='D'),
            'tipo': np.random.choice(['Receita', 'Despesa'], 90),
            'categoria': np.random.choice(['Vendas', 'Marketing', 'Salários', 'Impostos'], 90),
            'valor': np.random.uniform(100, 10000, 90).round(2),
            'status': np.random.choice(['Pago', 'Pendente', 'Atrasado'], 90)
        }
    
    df_example = pd.DataFrame(data)
    
    if st.button("📊 Usar este exemplo"):
        st.session_state.df = df_example
        st.session_state.data_loaded = True
        st.rerun()

with tab5:
    st.markdown("""
    ### 📖 Como usar este app
    
    1. **Conecte ao Ollama** na barra lateral
    2. **Carregue seus dados** em qualquer formato
    3. **Explore** visualizações e estatísticas
    4. **Faça perguntas** sobre seus dados
    
    ### 🎯 Dicas
    - Para arquivos grandes, use amostras primeiro
    - SQL funciona com queries SELECT apenas
    - O modelo entende português e inglês
    - Você pode fazer perguntas como:
        - "Qual a média de vendas?"
        - "Mostre tendências por categoria"
        - "Compare grupos diferentes"
    
    ### 🔧 Troubleshooting
    - Se o Ollama não conectar, verifique se está instalado
    - Formatos não suportados podem ser convertidos para CSV
    - Para arquivos muito grandes, o processamento pode ser lento
    """)

# ============================================
# SE OS DADOS FORAM CARREGADOS
# ============================================

if st.session_state.get('data_loaded', False):
    df = st.session_state.df
    
    # Métricas rápidas
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Linhas", f"{len(df):,}")
    with col2:
        st.metric("📋 Colunas", len(df.columns))
    with col3:
        st.metric("🔤 Tipos", len(df.dtypes.unique()))
    with col4:
        st.metric("📈 Memória", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Tabs para análise
    tab_vis, tab_stats, tab_qa, tab_raw = st.tabs(["📈 Visualizações", "📊 Estatísticas", "💬 Perguntas", "👀 Dados Brutos"])
    
    with tab_vis:
        st.markdown("### 📊 Visualizações Interativas")
        
        # Selecionar tipo de gráfico
        chart_type = st.selectbox(
            "Tipo de gráfico",
            ["📊 Barras", "📈 Linhas", "🥧 Pizza", "🔵 Dispersão", "📦 Boxplot", "🔥 Correlação"]
        )
        
        # Colunas numéricas e categóricas
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if chart_type == "📊 Barras":
            if cat_cols and num_cols:
                x_axis = st.selectbox("Eixo X (categorias)", cat_cols)
                y_axis = st.selectbox("Eixo Y (valores)", num_cols)
                
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} por {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados não têm colunas categóricas e numéricas suficientes")
        
        elif chart_type == "📈 Linhas":
            if df.select_dtypes(include=['datetime64']).columns.tolist():
                date_col = st.selectbox("Coluna de data", df.select_dtypes(include=['datetime64']).columns)
                y_axis = st.selectbox("Valor", num_cols)
                
                fig = px.line(df, x=date_col, y=y_axis, title=f"{y_axis} ao longo do tempo")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nenhuma coluna de data encontrada. Use gráfico de barras.")
        
        elif chart_type == "🔥 Correlação":
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Precisa de pelo menos 2 colunas numéricas")
    
    with tab_stats:
        st.markdown("### 📊 Estatísticas Descritivas")
        
        # Estatísticas gerais
        st.subheader("Resumo Numérico")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Valores nulos
        st.subheader("📉 Valores Ausentes")
        null_df = pd.DataFrame({
            'Coluna': df.columns,
            'Valores Nulos': df.isnull().sum().values,
            'Percentual': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(null_df, use_container_width=True)
        
        # Distribuição das colunas categóricas
        if cat_cols:
            st.subheader("📊 Distribuição Categórica")
            cat_col = st.selectbox("Selecione uma coluna categórica", cat_cols)
            
            freq_df = df[cat_col].value_counts().reset_index()
            freq_df.columns = [cat_col, 'Frequência']
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(freq_df, use_container_width=True)
            with col2:
                fig = px.pie(freq_df, values='Frequência', names=cat_col, title=f"Distribuição de {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab_qa:
        st.markdown("### 💬 Converse com seus Dados")
        
        if st.session_state.get('ollama_connected', False):
            # Inicializar chain se necessário
            if 'vector_store' not in st.session_state:
                with st.spinner("🔄 Preparando dados para consulta..."):
                    st.session_state.vector_store, _ = create_vector_store(
                        df, 
                        st.session_state.embeddings
                    )
                    
                    # Configurar prompt
                    template = """
                    Você é um analista de dados especialista.
                    Use APENAS as informações fornecidas abaixo para responder.
                    
                    INFORMAÇÕES RELEVANTES:
                    {context}
                    
                    PERGUNTA: {question}
                    
                    Responda de forma clara e objetiva em português.
                    Se a informação não estiver disponível, diga que não encontrou nos dados.
                    
                    RESPOSTA:
                    """
                    prompt = ChatPromptTemplate.from_template(template)
                    st.session_state.chain = prompt | st.session_state.llm | StrOutputParser()
            
            # Interface de perguntas
            col1, col2 = st.columns([3, 1])
            
            with col1:
                pergunta = st.text_input("❓ Faça sua pergunta:", placeholder="Ex: Qual a média de vendas?")
            
            with col2:
                num_results = st.number_input("Resultados", min_value=1, max_value=10, value=3)
            
            if pergunta:
                with st.spinner("🔍 Analisando..."):
                    resposta, docs = ask_question(
                        pergunta, 
                        st.session_state.vector_store,
                        st.session_state.llm,
                        st.session_state.chain
                    )
                    
                    # Mostrar resposta
                    st.markdown(f"**💡 Resposta:** {resposta}")
                    
                    # Mostrar fontes
                    with st.expander("📚 Ver fontes utilizadas"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Fonte {i}:**")
                            st.info(doc.page_content[:200] + "...")
            
            # Sugestões de perguntas
            with st.expander("💡 Sugestões de perguntas"):
                sugestoes = [
                    "Qual é o total de registros?",
                    "Mostre as estatísticas básicas",
                    "Existem valores ausentes?",
                    "Qual a distribuição dos dados?",
                    "Compare as categorias principais"
                ]
                for sug in sugestoes:
                    if st.button(sug, key=sug):
                        st.session_state.pergunta_sugerida = sug
                        st.rerun()
        else:
            st.warning("⚠️ Conecte ao Ollama na barra lateral primeiro!")
    
    with tab_raw:
        st.markdown("### 👀 Visualização dos Dados")
        
        # Opções de visualização
        rows_to_show = st.slider("Número de linhas", min_value=5, max_value=100, value=10)
        
        st.dataframe(df.head(rows_to_show), use_container_width=True)
        
        # Download dos dados processados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download como CSV",
            data=csv,
            file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # Mensagem inicial
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Bem-vindo ao Assistente IA Multi-Formatos!</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Para começar, carregue seus dados em uma das abas acima.
        </p>
        <p style="margin-top: 2rem;">
            Suportamos: CSV • Excel • JSON • TXT • SQL • URLs
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RODAPÉ
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Desenvolvido com ❤️ usando Streamlit, LangChain e Ollama
    </div>
    """,
    unsafe_allow_html=True
)