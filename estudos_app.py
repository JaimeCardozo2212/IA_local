import streamlit as st
import PyPDF2
import pdfplumber
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from datetime import datetime
import re

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="IA local proteção de dados sensíveis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILO CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #566573;
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
    .resumo-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F8F9F9;
        border-left: 5px solid #2E86C1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================

def extrair_texto_pdf(uploaded_file):
    """Extrai texto de arquivos PDF usando múltiplos métodos"""
    texto_completo = ""
    
    # Salvar arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Método 1: PyPDF2 (mais rápido, mas pode perder formatação)
        with open(tmp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                texto = page.extract_text()
                if texto:
                    texto_completo += texto + "\n"
        
        # Método 2: pdfplumber (melhor para tabelas e formatação)
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                texto = page.extract_text()
                if texto and len(texto) > len(texto_completo):
                    texto_completo = texto_completo + texto + "\n"
                
    except Exception as e:
        st.error(f"Erro ao ler PDF: {str(e)}")
    finally:
        # Limpar arquivo temporário
        os.unlink(tmp_path)
    
    # Limpar texto
    texto_completo = re.sub(r'\s+', ' ', texto_completo)  # Remove espaços extras
    texto_completo = re.sub(r'\n\s*\n', '\n\n', texto_completo)  # Remove linhas em branco
    
    return texto_completo

def extrair_texto_txt(uploaded_file):
    """Extrai texto de arquivos TXT"""
    try:
        texto = uploaded_file.read().decode('utf-8')
        return texto
    except:
        try:
            # Tentar outro encoding
            uploaded_file.seek(0)
            texto = uploaded_file.read().decode('latin-1')
            return texto
        except:
            return "Erro ao ler arquivo"

def dividir_em_chunks(texto, tamanho_chunk=1000, overlap=200):
    """Divide o texto em pedaços menores para processamento"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(texto)
    return chunks

def gerar_resumo(texto, llm):
    """Gera um resumo automático do texto"""
    prompt_resumo = """
    Crie um resumo estruturado do texto abaixo.
    
    Formato do resumo:
    1. TÓPICO PRINCIPAL: [Qual é o assunto principal?]
    2. PONTOS-CHAVE: [Liste os 5-7 pontos mais importantes]
    3. DETALHES RELEVANTES: [Informações específicas importantes]
    4. CONCLUSÃO: [Resumo final em 2-3 linhas]
    
    Texto para resumir:
    {texto}
    
    Resumo:
    """
    
    # Pegar apenas uma parte do texto para resumo (primeiros 5000 caracteres)
    texto_resumo = texto[:5000] + "..." if len(texto) > 5000 else texto
    
    resposta = llm.invoke(prompt_resumo.format(texto=texto_resumo))
    return resposta

# ============================================
# BARRA LATERAL - CONFIGURAÇÕES
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/reading--v1.png", width=100)
    st.markdown('<p class="main-header" style="font-size: 1.8rem;">📚 Dados sensíveis protegidos</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 💬 Assistente Rápido")

    with st.expander("🔍 Fazer pergunta geral"):
        pergunta_rapida = st.text_input("Pergunta:", placeholder="Digite algo...", key="pergunta_sidebar")
        
        if pergunta_rapida and st.button("Perguntar", key="btn_sidebar"):
            if st.session_state.get('ollama_connected', False):
                with st.spinner("Pensando..."):
                    llm = OllamaLLM(model=st.session_state.llm.model, temperature=0.7)
                    resposta = llm.invoke(f"Responda: {pergunta_rapida}")
                    st.info(resposta)
            else:
                st.warning("Conecte ao Ollama primeiro")
        st.divider()
    
    # Seleção do modelo
    st.markdown("### 🤖 Configurações da IA")
    modelo_llm = st.selectbox(
        "Modelo de Linguagem",
        ["llama3.2", "deepseek-r1", "mistral", "phi3"],
        index=0,
        help="Modelo que vai processar suas perguntas"
    )
    
    modelo_embedding = st.selectbox(
        "Modelo de Embedding",
        ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
        index=0,
        help="Modelo para buscar informações relevantes"
    )
    
    # Configurações de chunking
    st.markdown("### ⚙️ Configurações de Processamento")
    tamanho_chunk = st.slider(
        "Tamanho dos chunks (caracteres)", 
        min_value=500, 
        max_value=2000, 
        value=1000,
        step=100,
        help="Pedaços de texto para processar. Maior = mais contexto, menor = mais preciso"
    )
    
    overlap = st.slider(
        "Sobreposição entre chunks", 
        min_value=50, 
        max_value=500, 
        value=200,
        step=50,
        help="Sobreposição entre os pedaços para não perder contexto"
    )
    
    # Conexão com Ollama
    st.markdown("### 🔌 Status da IA")
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
        st.markdown('<div class="success-box">✅ IA conectada e pronta</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Conecte ao Ollama primeiro</div>', unsafe_allow_html=True)
        st.info("💡 Instale o Ollama em: https://ollama.com")
    
    st.divider()
    
    # Informações do app
    st.markdown("### 📖 Sobre")
    st.markdown("""
    Este app ajuda você a:
    - 📄 Estudar PDFs e textos
    - 📝 Gerar resumos automáticos
    - ❓ Responder perguntas do material
    - 🔍 Encontrar informações específicas
    
    **Como usar:**
    1. Conecte a IA
    2. Carregue seu material
    3. Faça perguntas ou peça resumos
    """)

# ============================================
# ÁREA PRINCIPAL
# ============================================
st.markdown('<p class="main-header">📚 IA local proteção de dados sensíveis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Carregue seus documentos sensíveis sem medo de vazamento</p>', unsafe_allow_html=True)

# ============================================
# UPLOAD DE ARQUIVOS
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "📤 Carregar documentos sensíveis",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Carregue PDFs ou arquivos de texto para estudar"
    )

with col2:
    st.markdown("### 📊 Estatísticas")
    if 'documentos' in st.session_state:
        total_chars = sum(len(doc.page_content) for doc in st.session_state.documentos)
        st.metric("📄 Documentos", len(st.session_state.documentos))
        st.metric("📝 Total caracteres", f"{total_chars:,}")
        st.metric("🧠 Chunks", len(st.session_state.get('chunks', [])))

# ============================================
# PROCESSAR ARQUIVOS CARREGADOS
# ============================================
if uploaded_files and st.session_state.get('ollama_connected', False):
    
    # Botão para processar
    if st.button("🔄 Processar Materiais", type="primary", use_container_width=True):
        with st.spinner("📖 Processando materiais... Isso pode levar alguns minutos"):
            
            todos_documentos = []
            todos_chunks = []
            
            # Barra de progresso
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Atualizar progresso
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.info(f"Processando: {uploaded_file.name}")
                
                # Extrair texto baseado no tipo
                if uploaded_file.type == "application/pdf":
                    texto = extrair_texto_pdf(uploaded_file)
                else:  # TXT
                    texto = extrair_texto_txt(uploaded_file)
                
                if texto and len(texto) > 100:  # Só processar se tiver texto
                    # Dividir em chunks
                    chunks = dividir_em_chunks(texto, tamanho_chunk, overlap)
                    
                    # Criar documentos para cada chunk
                    for j, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "fonte": uploaded_file.name,
                                "chunk": j,
                                "total_chunks": len(chunks)
                            }
                        )
                        todos_documentos.append(doc)
                        todos_chunks.append(chunk)
            
            if todos_documentos:
                # Criar banco vetorial
                st.session_state.vector_store = Chroma.from_documents(
                    documents=todos_documentos,
                    embedding=st.session_state.embeddings,
                    persist_directory="./chroma_estudos_db"
                )
                
                st.session_state.documentos = todos_documentos
                st.session_state.chunks = todos_chunks
                st.session_state.materiais_processados = True
                
                # Salvar texto completo para resumo
                texto_completo = " ".join(todos_chunks)
                st.session_state.texto_completo = texto_completo
                
                progress_bar.empty()
                st.markdown('<div class="success-box">✅ Materiais processados com sucesso!</div>', unsafe_allow_html=True)
                st.rerun()
            else:
                st.error("❌ Não foi possível extrair texto dos arquivos")

# ============================================
# INTERFACE PRINCIPAL (se materiais processados)
# ============================================
if st.session_state.get('materiais_processados', False):
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Resumo do Material", 
    "💬 Perguntas sobre o Material", 
    "📖 Visualizar Conteúdo", 
    "📚 Fontes",
    "🌐 Perguntas Gerais"  # NOVA ABA
])

    
    with tab1:
        st.markdown("### 📝 Resumo Automático do Material")
        st.markdown('<div class="info-box">📌 Clique no botão abaixo para gerar um resumo completo do material carregado</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 Gerar Resumo", use_container_width=True):
                with st.spinner("🤔 Analisando material e gerando resumo..."):
                    resumo = gerar_resumo(st.session_state.texto_completo, st.session_state.llm)
                    st.session_state.resumo_atual = resumo
        
        with col2:
            tipo_resumo = st.selectbox(
                "Tipo de resumo",
                ["Completo", "Pontos-chave apenas", "Perguntas para estudo"]
            )
        
        # Mostrar resumo se existir
        if st.session_state.get('resumo_atual'):
            st.markdown('<div class="resumo-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.resumo_atual)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Botão para copiar
            st.button("📋 Copiar resumo", key="copy_resumo")
    
    with tab2:
        st.markdown("### 💬 Faça Perguntas sobre o Material")
        st.markdown('<div class="info-box">📌 Pergunte qualquer coisa sobre o conteúdo carregado</div>', unsafe_allow_html=True)
        
        # Template para perguntas
        template_pergunta = """
        Você é um tutor especializado. Baseado APENAS no material fornecido, responda a pergunta do aluno.
        
        CONTEXTO (partes relevantes do material):
        {contexto}
        
        PERGUNTA DO ALUNO:
        {pergunta}
        
        Responda de forma clara e didática, como se estivesse explicando para um aluno.
        Se a resposta não estiver no material, diga que não encontrou essa informação.
        Use exemplos quando possível.
        
        RESPOSTA:
        """
        
        prompt = ChatPromptTemplate.from_template(template_pergunta)
        chain = prompt | st.session_state.llm | StrOutputParser()
        
        # Histórico de perguntas
        if 'historico_perguntas' not in st.session_state:
            st.session_state.historico_perguntas = []
        
        # Input da pergunta
        pergunta = st.text_input("❓ Sua pergunta:", placeholder="Ex: Qual é o conceito principal? Explique...")
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            n_resultados = st.number_input("Nº resultados", min_value=1, max_value=10, value=3)
        
        if pergunta:
            with st.spinner("🔍 Pesquisando nos materiais..."):
                # Buscar chunks relevantes
                docs_relevantes = st.session_state.vector_store.similarity_search(pergunta, k=n_resultados)
                
                # Preparar contexto
                contexto = "\n\n---\n\n".join([
                    f"[Fonte: {doc.metadata['fonte']} - Parte {doc.metadata['chunk'] + 1}/{doc.metadata['total_chunks']}]\n{doc.page_content}"
                    for doc in docs_relevantes
                ])
                
                # Gerar resposta
                resposta = chain.invoke({
                    "contexto": contexto,
                    "pergunta": pergunta
                })
                
                # Adicionar ao histórico
                st.session_state.historico_perguntas.append({
                    "pergunta": pergunta,
                    "resposta": resposta,
                    "fontes": [doc.metadata['fonte'] for doc in docs_relevantes]
                })
                
                # Mostrar resposta
                st.markdown("### 💡 Resposta:")
                st.markdown(resposta)
                
                # Mostrar fontes
                with st.expander("📚 Ver fontes consultadas"):
                    for i, doc in enumerate(docs_relevantes, 1):
                        st.markdown(f"**Fonte {i}: {doc.metadata['fonte']}**")
                        st.info(doc.page_content[:300] + "...")
        
        # Histórico de perguntas
        if st.session_state.historico_perguntas:
            with st.expander("📜 Histórico de perguntas"):
                for i, item in enumerate(reversed(st.session_state.historico_perguntas[-5:])):
                    st.markdown(f"**Pergunta:** {item['pergunta']}")
                    st.markdown(f"**Resposta:** {item['resposta'][:200]}...")
                    st.divider()
    
    with tab3:
        st.markdown("### 📖 Visualizar Conteúdo")
        
        # Selecionar fonte
        fontes = list(set([doc.metadata['fonte'] for doc in st.session_state.documentos]))
        fonte_selecionada = st.selectbox("Selecione o material", fontes)
        
        if fonte_selecionada:
            # Filtrar chunks da fonte selecionada
            chunks_fonte = [
                doc for doc in st.session_state.documentos 
                if doc.metadata['fonte'] == fonte_selecionada
            ]
            
            # Ordenar por número do chunk
            chunks_fonte.sort(key=lambda x: x.metadata['chunk'])
            
            # Mostrar conteúdo
            for chunk in chunks_fonte:
                with st.expander(f"Parte {chunk.metadata['chunk'] + 1} de {chunk.metadata['total_chunks']}"):
                    st.write(chunk.page_content)
    
    with tab4:
        st.markdown("### 📚 Materiais Carregados")
        
        # Listar todos os materiais
        for doc in st.session_state.documentos:
            if doc.metadata['chunk'] == 0:  # Mostrar apenas uma vez por arquivo
                st.markdown(f"""
                **📄 {doc.metadata['fonte']}**
                - Total de chunks: {doc.metadata['total_chunks']}
                - Tamanho aproximado: {len(doc.page_content) * doc.metadata['total_chunks']} caracteres
                """)
        
        # Estatísticas
        st.markdown("### 📊 Estatísticas Gerais")
        total_chars = sum(len(doc.page_content) for doc in st.session_state.documentos)
        total_palavras = total_chars / 5  # Aproximação
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de documentos", len(fontes))
        col2.metric("Total de chunks", len(st.session_state.documentos))
        col3.metric("Total de palavras", f"{int(total_palavras):,}")
    with tab5:
        st.markdown("### 🌐 Assistente para Perguntas Gerais")
        st.markdown('<div class="info-box">📌 Faça perguntas sobre QUALQUER assunto! Esta aba NÃO usa o material carregado.</div>', unsafe_allow_html=True)
        
        # Verificar se IA está conectada
        if not st.session_state.get('ollama_connected', False):
            st.warning("⚠️ Conecte ao Ollama na barra lateral primeiro!")
        else:
            # Layout em colunas
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Campo para pergunta geral
                pergunta_geral = st.text_area(
                    "💭 Digite sua pergunta:",
                    placeholder="Ex: Explique o conceito de gravidade quântica\nOu: Me ajude a criar uma receita de bolo\nOu: Qual a capital do Canadá?",
                    height=100
                )
            
            with col2:
                st.markdown("### ⚙️ Configurações")
                
                # Temperatura (criatividade)
                temperatura = st.slider(
                    "🎨 Criatividade",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="0 = mais preciso/factual, 1 = mais criativo/aleatório"
                )
                
                # Tamanho da resposta
                max_tokens = st.selectbox(
                    "📏 Tamanho da resposta",
                    [256, 512, 1024, 2048],
                    index=2,
                    help="Número máximo de tokens na resposta"
                )
                
                # Estilo de resposta
                estilo = st.selectbox(
                    "🎭 Estilo",
                    ["Padrão", "Profissional", "Didático", "Resumido", "Detalhado"]
                )
            
            # Botão para enviar
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                enviar = st.button("🚀 Enviar", type="primary", use_container_width=True)
            with col2:
                limpar = st.button("🗑️ Limpar", use_container_width=True)
            
            # Inicializar histórico de conversa geral
            if 'historico_geral' not in st.session_state:
                st.session_state.historico_geral = []
            
            if limpar:
                st.session_state.historico_geral = []
                st.session_state.ultima_resposta_geral = None
                st.rerun()
            
            if enviar and pergunta_geral:
                with st.spinner("🤔 Pensando..."):
                    try:
                        # Criar prompt baseado no estilo escolhido
                        if estilo == "Profissional":
                            prompt_estilo = "Responda de forma profissional, técnica e bem estruturada."
                        elif estilo == "Didático":
                            prompt_estilo = "Responda como um professor explicando para um aluno. Use analogias e exemplos simples."
                        elif estilo == "Resumido":
                            prompt_estilo = "Seja conciso. Responda em até 3 frases."
                        elif estilo == "Detalhado":
                            prompt_estilo = "Responda de forma detalhada, abrangendo todos os aspectos importantes."
                        else:  # Padrão
                            prompt_estilo = "Responda de forma clara e objetiva."
                        
                        # Montar prompt completo
                        prompt_completo = f"""
                        {prompt_estilo}
                        
                        Pergunta: {pergunta_geral}
                        
                        Resposta:
                        """
                        
                        # Configurar o modelo com temperatura
                        llm_temp = OllamaLLM(
                            model=st.session_state.llm.model,  # Usa mesmo modelo da sidebar
                            temperature=temperatura,
                            num_predict=max_tokens
                        )
                        
                        # Gerar resposta
                        resposta = llm_temp.invoke(prompt_completo)
                        
                        # Salvar no histórico
                        st.session_state.historico_geral.append({
                            "pergunta": pergunta_geral,
                            "resposta": resposta,
                            "estilo": estilo,
                            "temperatura": temperatura
                        })
                        
                        # Mostrar resposta
                        st.markdown("### 💡 Resposta:")
                        st.markdown(f'<div class="resumo-box">{resposta}</div>', unsafe_allow_html=True)
                        
                        # Botões de ação
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.button("📋 Copiar", key="copy_geral")
                        with col2:
                            st.button("🔄 Nova pergunta", key="new_question")
                        with col3:
                            st.button("👍 Útil", key="helpful")
                        
                    except Exception as e:
                        st.error(f"Erro ao gerar resposta: {str(e)}")
            
            # Mostrar histórico de conversas gerais
            if st.session_state.historico_geral:
                st.markdown("---")
                st.markdown("### 📜 Histórico de Conversas")
                
                for i, item in enumerate(reversed(st.session_state.historico_geral[-10:])):
                    with st.expander(f"❓ {item['pergunta'][:50]}..."):
                        st.markdown(f"**Pergunta:** {item['pergunta']}")
                        st.markdown(f"**Resposta:** {item['resposta']}")
                        st.caption(f"Estilo: {item['estilo']} | Temperatura: {item['temperatura']}")
            
            # Sugestões de perguntas
            st.markdown("---")
            st.markdown("### 💡 Sugestões de Perguntas")
            
            sugestoes = [
                "Explique a teoria da relatividade de forma simples",
                "Me ajude a criar um roteiro de estudos para Python",
                "Qual a diferença entre IA, Machine Learning e Deep Learning?",
                "Como funciona a internet? Explique para uma criança",
                "Me dê 5 dicas para aprender inglês mais rápido",
                "O que é blockchain e como funciona?",
                "Crie uma receita saudável para o café da manhã",
                "Qual a capital do Brasil e qual sua importância histórica?",
            ]
            
            # Criar botões em grid
            cols = st.columns(2)
            for i, sugestao in enumerate(sugestoes):
                with cols[i % 2]:
                    if st.button(sugestao, key=f"sugestao_{i}", use_container_width=True):
                        # Preencher o campo de pergunta com a sugestão
                        st.session_state.pergunta_sugerida = sugestao
                        st.rerun()

else:
    # Mensagem inicial se não houver materiais
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Bem-vindo ao seu Assistente!</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Para começar, siga estes passos:
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">1️⃣</div>
                <p>Conecte a IA na barra lateral</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem;">2️⃣</div>
                <p>Carregue seus PDFs ou textos</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem;">3️⃣</div>
                <p>Faça perguntas e peça resumos</p>
            </div>
        </div>
        <p style="color: #888;">
            📚 Suporta PDFs e arquivos de texto
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
        Assistente de IA com segurança | Desenvolvido com Streamlit + LangChain + Ollama
    </div>
    """,
    unsafe_allow_html=True
)