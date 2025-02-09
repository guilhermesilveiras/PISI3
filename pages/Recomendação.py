import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.title("🏙️ Sistema de Recomendação de Cidades Similares utlizando DBSCAN e KNN")

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv("data.csv").dropna(subset=['city', 'country'])

data = load_data()

# Configuração
FEATURES_RECOMENDACAO = [
    'x48', 'x49', 'x54', 'x1', 'x2', 'x3', 'x28', 'x33', 'x41'
]
DESCRICOES = {
    'x48': 'Aluguel Centro',
    'x49': 'Aluguel Fora do Centro',
    'x54': 'Salário Médio',
    'x1': 'Refeição Barata',
    'x2': 'Jantar para Dois',
    'x3': 'McMeal',
    'x28': 'Transporte Público',
    'x33': 'Gasolina',
    'x41': 'Cinema'
}

# Pré-processamento
def preprocessar_dados(df, features):
    data_clean = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    return scaler.fit_transform(data_clean)

dados_padronizados = preprocessar_dados(data, FEATURES_RECOMENDACAO)

# Clusterização Híbrida (DBSCAN + k-NN)
def recomendar_hibrido(dados, cidade_idx, num_recomendacoes=5):
    # Passo 1: Clusterização com DBSCAN
    clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(dados)
    cluster_alvo = clusters[cidade_idx]
    
    # Passo 2: Filtrar cidades do mesmo cluster
    mascara_cluster = (clusters == cluster_alvo)
    dados_cluster = dados[mascara_cluster]
    
    # Passo 3: k-NN dentro do cluster
    nn_model = NearestNeighbors(n_neighbors=num_recomendacoes+1, metric='cosine')
    nn_model.fit(dados_cluster)
    distances, indices = nn_model.kneighbors([dados[cidade_idx]])
    
    # Retornar índices das cidades recomendadas
    return indices[0][1:]

# Interface
tipo_recomendacao = st.radio("Escolha o tipo de recomendação:", ['Global', 'Por País'])

if tipo_recomendacao == 'Por País':
    pais_selecionado = st.selectbox("Selecione um país:", options=data['country'].unique())
    dados_filtrados = data[data['country'] == pais_selecionado]
    max_cidades = len(dados_filtrados) - 1
else:
    pais_selecionado = None
    dados_filtrados = data
    max_cidades = len(data) - 1

num_recomendacoes = st.slider("Número de recomendações:", 
                             min_value=1, 
                             max_value=min(10, max_cidades), 
                             value=min(5, max_cidades))

cidade_selecionada = st.selectbox("Selecione uma cidade para comparação:", 
                                 options=dados_filtrados['city'].unique())

if st.button("Buscar Cidades Similares"):
    # Encontrar índice da cidade alvo
    idx = dados_filtrados[dados_filtrados['city'] == cidade_selecionada].index[0]
    
    # Recomendar cidades
    indices_recomendadas = recomendar_hibrido(dados_padronizados, idx, num_recomendacoes)
    recomendadas = data.iloc[indices_recomendadas]
    
    # Visualização t-SNE
    st.subheader("Visualização t-SNE")
    tsne = TSNE(n_components=2, perplexity=30)
    dados_tsne = tsne.fit_transform(dados_padronizados)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(dados_tsne[:,0], dados_tsne[:,1], alpha=0.3, label='Todas cidades')
    ax.scatter(dados_tsne[indices_recomendadas,0], dados_tsne[indices_recomendadas,1], 
               color='red', label='Recomendadas')
    ax.scatter(dados_tsne[idx,0], dados_tsne[idx,1], 
               color='black', marker='*', s=200, label='Alvo')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.legend()
    st.pyplot(fig)
    
    # Tabela comparativa
    st.subheader("Comparação Detalhada")
    cols = ['city', 'country'] + FEATURES_RECOMENDACAO
    comparacao = pd.concat([data.iloc[[idx]], recomendadas])[cols]
    comparacao = comparacao.rename(columns=DESCRICOES)
    st.dataframe(comparacao.style.format({col: "{:.2f}" for col in DESCRICOES.values()}))

    st.write("recomendação com base em cidades com perfis de custo de vida semelhantes, utilizando as estrategias DBSCAN (Density-Based Spatial Clustering) para identificar grupos naturais de cidades com características similares (proporção em relação aos custos) e K-NN k-Nearest Neighbors: Para encontrar as cidades mais próximas dentro desses grupos")