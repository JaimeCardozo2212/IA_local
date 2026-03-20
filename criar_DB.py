import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ============================================
# CONFIGURAÇÃO DOS DADOS
# ============================================
print("🔄 Gerando base de dados de teste...")

# Configurar seed para reproducibilidade
np.random.seed(42)
random.seed(42)

# ============================================
# LISTAS DE DADOS
# ============================================

# Produtos por categoria
produtos = {
    'Eletrônicos': [
        ('Smartphone XYZ', 2500, 1800),
        ('Notebook Pro', 5200, 4100),
        ('Tablet 10"', 1800, 1300),
        ('Fone Bluetooth', 350, 220),
        ('Smartwatch', 890, 600),
        ('Carregador Portátil', 150, 90),
        ('Caixa de Som', 420, 280),
        ('Monitor 24"', 1300, 950),
        ('Mouse Gamer', 280, 180),
        ('Teclado Mecânico', 450, 310)
    ],
    'Vestuário': [
        ('Camiseta Básica', 80, 35),
        ('Calça Jeans', 180, 95),
        ('Jaqueta Corta-Vento', 250, 140),
        ('Tênis Esportivo', 320, 190),
        ('Boné', 60, 25),
        ('Meia 3 Pares', 35, 15),
        ('Camisa Social', 150, 75),
        ('Short Esportivo', 90, 45),
        ('Vestido', 200, 110),
        ('Blusa de Frio', 160, 85)
    ],
    'Alimentos': [
        ('Café Gourmet 500g', 45, 28),
        ('Chocolate Importado', 25, 14),
        ('Azeite Extra Virgem', 55, 35),
        ('Vinho Tinto', 120, 75),
        ('Queijo Especial', 65, 40),
        ('Geleia Artesanal', 28, 15),
        ('Biscoito Importado', 18, 9),
        ('Granola Premium', 32, 18),
        ('Chá Especial', 22, 12),
        ('Mel Puro', 48, 28)
    ],
    'Casa e Decoração': [
        ('Jogo de Lençol', 180, 95),
        ('Toalha de Banho', 70, 35),
        ('Tapete', 320, 180),
        ('Abajur', 150, 80),
        ('Quadro Decorativo', 220, 110),
        ('Vaso Cerâmica', 90, 40),
        ('Almofada', 65, 30),
        ('Cortina', 140, 75),
        ('Espelho', 280, 150),
        ('Luminária', 110, 55)
    ],
    'Livros': [
        ('Best Seller Romance', 60, 35),
        ('Livro Técnico Python', 120, 75),
        ('Box Trilogia', 150, 90),
        ('Autoajuda', 45, 25),
        ('Biografia', 55, 32),
        ('Livro Infantil', 40, 22),
        ('Atlas Geográfico', 180, 110),
        ('Dicionário', 90, 50),
        ('Livro de Receitas', 70, 38),
        ('HQs', 35, 18)
    ]
}

# Clientes
nomes = [
    'João Silva', 'Maria Santos', 'Pedro Oliveira', 'Ana Souza', 
    'Carlos Lima', 'Fernanda Alves', 'Rafael Costa', 'Juliana Ferreira',
    'Marcos Pereira', 'Patrícia Gomes', 'Lucas Rodrigues', 'Camila Martins',
    'Bruno Carvalho', 'Amanda Nunes', 'Diego Barbosa', 'Larissa Ribeiro',
    'Thiago Mendes', 'Vanessa Araújo', 'Rodrigo Cardoso', 'Gabriela Teixeira'
]

cidades = [
    ('São Paulo', 'SP'), ('Rio de Janeiro', 'RJ'), ('Belo Horizonte', 'MG'),
    ('Brasília', 'DF'), ('Salvador', 'BA'), ('Fortaleza', 'CE'),
    ('Curitiba', 'PR'), ('Manaus', 'AM'), ('Recife', 'PE'), ('Porto Alegre', 'RS'),
    ('Goiânia', 'GO'), ('Belém', 'PA'), ('Guarulhos', 'SP'), ('Campinas', 'SP'),
    ('São Luís', 'MA'), ('São Gonçalo', 'RJ'), ('Maceió', 'AL'), ('Duque de Caxias', 'RJ'),
    ('Natal', 'RN'), ('Teresina', 'PI'), ('São Bernardo do Campo', 'SP'),
    ('Campo Grande', 'MS'), ('Jaboatão dos Guararapes', 'PE'), ('Osasco', 'SP')
]

# Formas de pagamento
pagamentos = ['Cartão Crédito', 'Cartão Débito', 'PIX', 'Boleto', 'Dinheiro', 'Transferência']

# Status do pedido
status = ['Entregue', 'Enviado', 'Processando', 'Cancelado', 'Atrasado']

# Vendedores
vendedores = [
    'Carlos Vendas', 'Mariana Oliveira', 'Roberto Santos', 'Patrícia Lima',
    'André Costa', 'Cristina Rocha', 'Fernando Alves', 'Juliana Mendes'
]

# ============================================
# GERAR DADOS DE VENDAS
# ============================================

dados_vendas = []
data_inicio = datetime(2024, 1, 1)
data_fim = datetime(2024, 12, 31)
dias_totais = (data_fim - data_inicio).days

for i in range(500):  # 500 vendas
    # Selecionar categoria e produto aleatoriamente
    categoria = random.choice(list(produtos.keys()))
    produto_info = random.choice(produtos[categoria])
    produto, preco_venda, preco_custo = produto_info
    
    # Quantidade (com viés para quantidades menores)
    quantidade = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
    
    # Calcular valores
    valor_total = preco_venda * quantidade
    custo_total = preco_custo * quantidade
    lucro = valor_total - custo_total
    margem = (lucro / valor_total * 100).round(1)
    
    # Data aleatória dentro de 2024
    dias_aleatorios = random.randint(0, dias_totais)
    data_venda = data_inicio + timedelta(days=dias_aleatorios)
    
    # Cliente aleatório
    cliente = random.choice(nomes)
    
    # Cidade e estado
    cidade, estado = random.choice(cidades)
    
    # Forma de pagamento
    pagamento = random.choice(pagamentos)
    
    # Status (com probabilidades diferentes)
    status_venda = np.random.choice(
        status, 
        p=[0.6, 0.2, 0.1, 0.05, 0.05]  # 60% entregue, 20% enviado, etc.
    )
    
    # Vendedor
    vendedor = random.choice(vendedores)
    
    # Nota do cliente (1-5) - apenas para entregues
    if status_venda == 'Entregue':
        nota = np.random.choice([3, 4, 4, 5, 5, 5], p=[0.05, 0.15, 0.2, 0.3, 0.2, 0.1])
    else:
        nota = None
    
    # Adicionar registro
    dados_vendas.append({
        'ID_Venda': f'V{str(i+1).zfill(4)}',
        'Data': data_venda.strftime('%Y-%m-%d'),
        'Cliente': cliente,
        'Cidade': cidade,
        'Estado': estado,
        'Categoria': categoria,
        'Produto': produto,
        'Preco_Unitario': preco_venda,
        'Quantidade': quantidade,
        'Valor_Total': valor_total,
        'Custo_Total': custo_total,
        'Lucro': lucro,
        'Margem_%': margem,
        'Forma_Pagamento': pagamento,
        'Status': status_venda,
        'Vendedor': vendedor,
        'Nota_Cliente': nota,
        'Regiao': np.random.choice(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'], p=[0.1, 0.15, 0.1, 0.45, 0.2])
    })

# ============================================
# CRIAR DATAFRAME
# ============================================

df_vendas = pd.DataFrame(dados_vendas)

# ============================================
# ADICIONAR MAIS UMA ABA COM RESUMO
# ============================================

# Criar resumo mensal
df_vendas['Data'] = pd.to_datetime(df_vendas['Data'])
df_vendas['Mes'] = df_vendas['Data'].dt.month
df_vendas['Ano'] = df_vendas['Data'].dt.year
df_vendas['Mes_Ano'] = df_vendas['Data'].dt.strftime('%Y-%m')

resumo_mensal = df_vendas.groupby('Mes_Ano').agg({
    'ID_Venda': 'count',
    'Valor_Total': 'sum',
    'Lucro': 'sum',
    'Nota_Cliente': 'mean'
}).round(2).reset_index()

resumo_mensal.columns = ['Mês/Ano', 'Qtd_Vendas', 'Faturamento', 'Lucro_Total', 'Nota_Média']

resumo_categorias = df_vendas.groupby('Categoria').agg({
    'ID_Venda': 'count',
    'Valor_Total': 'sum',
    'Lucro': 'sum',
    'Margem_%': 'mean'
}).round(2).reset_index()

resumo_categorias.columns = ['Categoria', 'Qtd_Vendas', 'Faturamento', 'Lucro', 'Margem_Média_%']

# ============================================
# SALVAR ARQUIVO EXCEL COM MÚLTIPLAS ABAS
# ============================================

nome_arquivo = 'dados_teste_vendas.xlsx'

with pd.ExcelWriter(nome_arquivo, engine='xlsxwriter') as writer:
    # Aba principal - Vendas
    df_vendas.to_excel(writer, sheet_name='Vendas_Completas', index=False)
    
    # Aba de resumo mensal
    resumo_mensal.to_excel(writer, sheet_name='Resumo_Mensal', index=False)
    
    # Aba de resumo por categoria
    resumo_categorias.to_excel(writer, sheet_name='Resumo_Categorias', index=False)
    
    # Aba com informações do dataset
    info_df = pd.DataFrame({
        'Informação': [
            'Total de Registros',
            'Período Início',
            'Período Fim',
            'Número de Clientes',
            'Número de Produtos',
            'Faturamento Total',
            'Lucro Total',
            'Ticket Médio',
            'Margem Média'
        ],
        'Valor': [
            len(df_vendas),
            df_vendas['Data'].min().strftime('%d/%m/%Y'),
            df_vendas['Data'].max().strftime('%d/%m/%Y'),
            df_vendas['Cliente'].nunique(),
            df_vendas['Produto'].nunique(),
            f"R$ {df_vendas['Valor_Total'].sum():,.2f}",
            f"R$ {df_vendas['Lucro'].sum():,.2f}",
            f"R$ {df_vendas['Valor_Total'].mean():,.2f}",
            f"{df_vendas['Margem_%'].mean():.1f}%"
        ]
    })
    info_df.to_excel(writer, sheet_name='Info_Geral', index=False)
    
    # Formatação das planilhas
    workbook = writer.book
    for sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
        
        # Formato para valores monetários
        if sheet in ['Vendas_Completas', 'Resumo_Mensal', 'Resumo_Categorias']:
            # Ajustar largura das colunas automaticamente
            worksheet.set_column('A:Z', 15)

print(f"✅ Arquivo '{nome_arquivo}' criado com sucesso!")
print(f"📊 Total de registros: {len(df_vendas)}")
print(f"💰 Faturamento total: R$ {df_vendas['Valor_Total'].sum():,.2f}")
print(f"📈 Lucro total: R$ {df_vendas['Lucro'].sum():,.2f}")
print(f"📁 Arquivo salvo em: {os.path.abspath(nome_arquivo)}")

# Mostrar primeiras linhas
print("\n👀 Primeiras 5 vendas:")
print(df_vendas[['Data', 'Cliente', 'Categoria', 'Produto', 'Valor_Total', 'Status']].head().to_string())

# ============================================
# GERAR TAMBÉM VERSÃO CSV
# ============================================
csv_nome = 'dados_teste_vendas.csv'
df_vendas.to_csv(csv_nome, index=False, encoding='utf-8-sig')
print(f"\n✅ Versão CSV também gerada: {csv_nome}")