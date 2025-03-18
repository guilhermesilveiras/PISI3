import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.title("🏙️ Sistema de Recomendação de Cidades Similares")

# ==============================
# Carregar Dados
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("data_cleaned.csv").dropna(subset=['city', 'country'])

data = load_data()

# ==============================
# Configurações Globais
# ==============================
FEATURES = ['x48', 'x49', 'x54', 'x1', 'x2', 'x3', 'x28', 'x33', 'x41']
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

# ==============================
# Funções Auxiliares
# ==============================
def preprocessar_dados(df, features):
    """Normaliza os dados selecionados."""
    df_clean = df[features].fillna(df[features].median())
    return StandardScaler().fit_transform(df_clean)

def calcular_eps(dados, k=5):
    """Calcula o valor ideal de eps para DBSCAN baseado na distância média dos k vizinhos."""
    nn = NearestNeighbors(n_neighbors=k+1).fit(dados)
    distances, _ = nn.kneighbors(dados)
    return np.percentile(distances[:, -1], 80)

def recomendar_knn(cluster_labels, dados, indices, cidade_idx, num_recomendacoes=5):
    """Recomenda cidades dentro do mesmo cluster usando k-NN."""
    cluster_alvo = cluster_labels[cidade_idx]
    if cluster_alvo == -1:
        return np.array([])
    indices_cluster = indices[cluster_labels == cluster_alvo]
    dados_cluster = dados[cluster_labels == cluster_alvo]
    nn_model = NearestNeighbors(n_neighbors=min(num_recomendacoes+1, len(dados_cluster)), metric='cosine').fit(dados_cluster)
    idx_local = list(indices_cluster).index(cidade_idx)
    _, indices_knn = nn_model.kneighbors([dados_cluster[idx_local]])
    return indices_cluster[indices_knn[0][1:]]

# ==============================
# Processamento Inicial
# ==============================
dados_padronizados = preprocessar_dados(data, FEATURES)
eps_otimizado = calcular_eps(dados_padronizados)

# ==============================
# Interface do Usuário
# ==============================
algoritmo = st.radio("Selecione o algoritmo de clusterização:", ["DBSCAN", "HDBSCAN"])

if algoritmo == "DBSCAN":
    eps_val = st.slider("Valor de EPS:", 0.1, 5.0, float(eps_otimizado))
    min_samples = st.slider("Min Samples:", 1, 20, 5)
else:
    min_cluster_size = st.slider("Tamanho mínimo do cluster:", 2, 20, 5)
    min_samples = st.slider("Min Samples:", 1, 20, 5)

tipo_recomendacao = st.radio("Escolha o tipo de recomendação:", ['Global', 'Por País'])
dados_filtrados = data if tipo_recomendacao == 'Global' else data[data['country'] == st.selectbox("Selecione um país:", data['country'].unique())]

if len(dados_filtrados) >= 2:
    num_recomendacoes = st.slider("Número de recomendações:", 1, min(10, len(dados_filtrados)-1), 5)
    cidade_selecionada = st.selectbox("Selecione uma cidade:", dados_filtrados['city'].unique())

    if st.button("Buscar Cidades Similares"):
        idx = dados_filtrados[dados_filtrados['city'] == cidade_selecionada].index[0]
        indices_filtrados = dados_filtrados.index.to_numpy()
        dados_filtrados_std = dados_padronizados[indices_filtrados, :]
        
        if algoritmo == "DBSCAN":
            clusters = DBSCAN(eps=eps_val, min_samples=min_samples).fit_predict(dados_filtrados_std)
        else:
            clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(dados_filtrados_std)
        
        indices_recomendadas = recomendar_knn(clusters, dados_filtrados_std, indices_filtrados, idx, num_recomendacoes)

        if len(indices_recomendadas) == 0:
            st.warning("Nenhuma cidade recomendada encontrada no mesmo cluster.")
        else:
            recomendadas = data.loc[indices_recomendadas]
            comparacao = pd.concat([data.loc[[idx]], recomendadas])[['city', 'country'] + FEATURES].rename(columns=DESCRICOES)
            st.subheader("Comparação Detalhada")
            st.dataframe(comparacao.style.format({col: "{:.2f}" for col in DESCRICOES.values()}))

            # Visualização t-SNE
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            dados_tsne = tsne.fit_transform(dados_filtrados_std)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(dados_tsne[:, 0], dados_tsne[:, 1], alpha=0.3, label='Cidades')
            idx_tsne = list(dados_filtrados.index).index(idx)
            ax.scatter(dados_tsne[idx_tsne, 0], dados_tsne[idx_tsne, 1], color='black', marker='*', s=200, label='Alvo')
            recomendadas_tsne = [list(dados_filtrados.index).index(i) for i in indices_recomendadas]
            ax.scatter(dados_tsne[recomendadas_tsne, 0], dados_tsne[recomendadas_tsne, 1], color='red', label='Recomendadas')
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.legend()
            st.pyplot(fig)
