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

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv("data_cleaned.csv").dropna(subset=['city', 'country'])

data = load_data()

# ==============================
# Configuração
# ==============================
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

# ==============================
# Funções Auxiliares
# ==============================
def calcular_eps(dados, k=5):
    """Calcula o valor ideal de eps para DBSCAN baseado na distância média dos k vizinhos."""
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(dados)
    distances, _ = nn.kneighbors(dados)
    return np.percentile(distances[:, -1], 80)

# ==============================
# Pré-processamento
# ==============================
def preprocessar_dados(df, features):
    data_clean = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    return scaler.fit_transform(data_clean)

dados_padronizados = preprocessar_dados(data, FEATURES_RECOMENDACAO)
eps_otimizado = calcular_eps(dados_padronizados, k=5)

# ==============================
# Funções de Recomendação
# ==============================
def recomendar_dbscan_knn(dados, indices_filtrados, cidade_idx, num_recomendacoes=5, eps=0.5, min_samples=5):
    dados_filtrados = dados[indices_filtrados, :]
    idx_local = list(indices_filtrados).index(cidade_idx)
    
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(dados_filtrados)
    cluster_alvo = clusters[idx_local]
    
    if cluster_alvo == -1:  # Se for ruído
        return np.array([])
    
    mascara_cluster = (clusters == cluster_alvo)
    dados_cluster = dados_filtrados[mascara_cluster]
    indices_cluster = indices_filtrados[mascara_cluster]
    
    nn_model = NearestNeighbors(n_neighbors=min(num_recomendacoes+1, len(dados_cluster)), metric='cosine')
    nn_model.fit(dados_cluster)
    idx_local_cluster = list(indices_cluster).index(cidade_idx)
    distances, indices = nn_model.kneighbors([dados_cluster[idx_local_cluster]])
    
    return indices_cluster[indices[0][1:]]

def recomendar_hdbscan_knn(dados, indices_filtrados, cidade_idx, num_recomendacoes=5, 
                          min_cluster_size=5, min_samples=None):
    dados_filtrados = dados[indices_filtrados, :]
    idx_local = list(indices_filtrados).index(cidade_idx)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples,
                                gen_min_span_tree=True)
    clusters = clusterer.fit_predict(dados_filtrados)
    cluster_alvo = clusters[idx_local]
    
    if cluster_alvo == -1:  # Se for ruído
        return np.array([])
    
    mascara_cluster = (clusters == cluster_alvo)
    dados_cluster = dados_filtrados[mascara_cluster]
    indices_cluster = indices_filtrados[mascara_cluster]
    
    nn_model = NearestNeighbors(n_neighbors=min(num_recomendacoes+1, len(dados_cluster)), metric='cosine')
    nn_model.fit(dados_cluster)
    idx_local_cluster = list(indices_cluster).index(cidade_idx)
    distances, indices = nn_model.kneighbors([dados_cluster[idx_local_cluster]])
    
    return indices_cluster[indices[0][1:]]

# ==============================
# Interface do Usuário
# ==============================
algoritmo = st.radio("Selecione o algoritmo de clusterização:", ["DBSCAN + k-NN", "HDBSCAN + k-NN"])

# Definir parâmetros com valores padrão
eps_val = eps_otimizado  # Valor padrão para DBSCAN
min_samples_dbscan = 5    # Valor padrão para DBSCAN
min_cluster_size = 5      # Valor padrão para HDBSCAN
min_samples_hdbscan = 5   # Valor padrão para HDBSCAN

if algoritmo == "DBSCAN + k-NN":
    eps_val = st.slider("Valor de EPS para DBSCAN:", 0.1, 5.0, float(eps_otimizado))
    min_samples_dbscan = st.slider("Min Samples para DBSCAN:", 1, 20, 5)
else:
    min_cluster_size = st.slider("Tamanho mínimo do cluster:", 2, 20, 5)
    min_samples_hdbscan = st.slider("Min Samples para HDBSCAN:", 1, 20, 5)

tipo_recomendacao = st.radio("Escolha o tipo de recomendação:", ['Global', 'Por País'])

if tipo_recomendacao == 'Por País':
    pais_selecionado = st.selectbox("Selecione um país:", options=data['country'].unique())
    dados_filtrados = data[data['country'] == pais_selecionado]
else:
    dados_filtrados = data

if len(dados_filtrados) < 2:
    st.error("Número insuficiente de cidades no filtro selecionado para gerar recomendações.")
else:
    num_recomendacoes = st.slider("Número de recomendações:", 
                                 min_value=1, 
                                 max_value=min(10, len(dados_filtrados)-1), 
                                 value=min(5, len(dados_filtrados)-1))
    
    cidade_selecionada = st.selectbox("Selecione uma cidade para comparação:", 
                                     options=dados_filtrados['city'].unique())

    if st.button("Buscar Cidades Similares"):
        idx = dados_filtrados[dados_filtrados['city'] == cidade_selecionada].index[0]
        
        if algoritmo == "DBSCAN + k-NN":
            indices_recomendadas = recomendar_dbscan_knn(dados_padronizados, dados_filtrados.index.to_numpy(), idx, 
                                                         num_recomendacoes, eps_val, min_samples_dbscan)
            st.write(f"**DBSCAN** executado com eps={eps_val:.2f}, min_samples={min_samples_dbscan}")
        else:
            indices_recomendadas = recomendar_hdbscan_knn(dados_padronizados, dados_filtrados.index.to_numpy(), idx,
                                                          num_recomendacoes, min_cluster_size, min_samples_hdbscan)
            st.write(f"**HDBSCAN** executado com min_cluster_size={min_cluster_size}, min_samples={min_samples_hdbscan}")
        
        if len(indices_recomendadas) == 0:
            st.warning("Nenhuma cidade recomendada encontrada no mesmo cluster.")
        else:
            recomendadas = data.loc[indices_recomendadas]
            
            # Visualização t-SNE
            st.subheader("Visualização t-SNE")
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            dados_tsne = tsne.fit_transform(dados_padronizados[dados_filtrados.index, :])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(dados_tsne[:, 0], dados_tsne[:, 1], alpha=0.3, label='Cidades filtradas')
            idx_tsne = list(dados_filtrados.index).index(idx)
            ax.scatter(dados_tsne[idx_tsne, 0], dados_tsne[idx_tsne, 1], color='black', marker='*', s=200, label='Alvo')
            recomendadas_tsne = [list(dados_filtrados.index).index(i) for i in indices_recomendadas]
            ax.scatter(dados_tsne[recomendadas_tsne, 0], dados_tsne[recomendadas_tsne, 1], color='red', label='Recomendadas')
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.legend()
            st.pyplot(fig)
            
            # Tabela comparativa
            st.subheader("Comparação Detalhada")
            cols = ['city', 'country'] + FEATURES_RECOMENDACAO
            comparacao = pd.concat([data.loc[[idx]], recomendadas])[cols]
            comparacao = comparacao.rename(columns=DESCRICOES)
            st.dataframe(comparacao.style.format({col: "{:.2f}" for col in DESCRICOES.values()}))
            
            st.write("""
            **Recomendação** baseada em cidades com perfis de custo de vida semelhantes, 
            combinando **clusterização** (DBSCAN ou HDBSCAN) e **k-NN** 
            para encontrar as cidades mais próximas dentro de cada agrupamento.
            """)
            
            dados_filtrados_std = dados_padronizados[dados_filtrados.index, :]

            # --- DBSCAN ---
            labels_dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_dbscan).fit_predict(dados_filtrados_std)
            silhouette_dbscan = (silhouette_score(dados_filtrados_std[labels_dbscan != -1], labels_dbscan[labels_dbscan != -1])
                                 if len(np.unique(labels_dbscan[labels_dbscan != -1])) > 1 else np.nan)

            # --- HDBSCAN ---
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples_hdbscan)
            labels_hdbscan = clusterer.fit_predict(dados_filtrados_std)
            silhouette_hdbscan = (silhouette_score(dados_filtrados_std[labels_hdbscan != -1], labels_hdbscan[labels_hdbscan != -1])
                                  if len(np.unique(labels_hdbscan[labels_hdbscan != -1])) > 1 else np.nan)

            st.write("### Acurácia dos Clusters")
            st.write(f"**DBSCAN:** {silhouette_dbscan:.2f}" if not np.isnan(silhouette_dbscan) else "**DBSCAN:** Não aplicável")
            st.write(f"**HDBSCAN:** {silhouette_hdbscan:.2f}" if not np.isnan(silhouette_hdbscan) else "**HDBSCAN:** Não aplicável")

            # ==============================
            # Gráfico de Comparação das Métricas
            # ==============================
            fig2, ax2 = plt.subplots()
            algoritmos = ['DBSCAN', 'HDBSCAN']
            scores = [silhouette_dbscan, silhouette_hdbscan]
            cores = ['blue', 'green']
            ax2.bar(algoritmos, [s if not np.isnan(s) else 0 for s in scores], color=cores)
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Comparação de Acurácia dos Clusters')
            st.pyplot(fig2)

            # ==============================
            # Gráfico de t-SNE Colorido pelos Clusters
            # ==============================
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            dados_tsne = tsne.fit_transform(dados_filtrados_std)

            fig3, axs = plt.subplots(1, 2, figsize=(14, 6))
            sc1 = axs[0].scatter(dados_tsne[:, 0], dados_tsne[:, 1], c=labels_dbscan, cmap='viridis', alpha=0.6)
            axs[0].set_title("t-SNE com Clusters - DBSCAN")
            fig3.colorbar(sc1, ax=axs[0])
            sc2 = axs[1].scatter(dados_tsne[:, 0], dados_tsne[:, 1], c=labels_hdbscan, cmap='viridis', alpha=0.6)
            axs[1].set_title("t-SNE com Clusters - HDBSCAN")
            fig3.colorbar(sc2, ax=axs[1])
            st.pyplot(fig3)