import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Dicionário com nomes completos das features
FEATURE_DICT = {
    'x1': 'Refeição em Restaurante Barato (USD)',
    'x2': 'Jantar para Dois em Restaurante Médio (USD)',
    'x3': 'Combo no McDonalds (USD)',
    'x4': 'Cerveja Nacional (Bar) (USD)',
    'x27': 'Cigarros (Marlboro) (USD)',
    'x28': 'Passagem de Ônibus (USD)',
    'x31': 'Corrida de Táxi por km (USD)',
    'x33': 'Gasolina (1 litro) (USD)',
    'x36': 'Contas Básicas (Eletricidade, Água, etc.) (USD)',
    'x38': 'Internet (60Mbps) (USD)',
    'x41': 'Cinema (1 ingresso) (USD)',
    'x48': 'Aluguel Centro (1 quarto) (USD)',
    'x49': 'Aluguel Fora do Centro (1 quarto) (USD)',
    'x54': 'Salário Médio Líquido (USD)'
}

SELECTABLE_FEATURES = [
    'x54', 'x49', 'x48', 'x41', 'x38', 'x36',
    'x4', 'x3', 'x2', 'x1', 'x27', 'x28', 'x31', 'x33'
]

st.title("Análise de Cluster de Cidades por Custo de Vida")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_cleaned.csv")
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("Arquivo 'data_cleaned.csv' não encontrado!")
        return None

data = load_data()

if data is not None:
    selected_features = st.multiselect(
        "Selecione as características para clusterização:",
        options=SELECTABLE_FEATURES,
        format_func=lambda x: FEATURE_DICT[x],
        default=['x54', 'x48', 'x41', 'x38', 'x28']
    )
    
    if not selected_features:
        st.warning("Selecione pelo menos uma característica!")
    else:
        k = st.slider("Número de Clusters", 2, 10, 5)
        
        # Pré-processamento
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[selected_features])
        
        # Redução de dimensionalidade para visualização com PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        
        # Clusterização com KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        data['Cluster'] = clusters
        
        # Cálculo do Silhouette Score
        sil_score = silhouette_score(data_scaled, clusters)
        st.write(f"**Silhouette Score para k = {k}: {sil_score:.2f}**")
        
        # Gráfico de Silhueta para o número de clusters selecionado
        silhouette_vals = silhouette_samples(data_scaled, clusters)
        fig_sil, ax_sil = plt.subplots(figsize=(10, 6))
        y_lower = 10
        for i in range(k):
            # Valores de silhueta para o cluster i, ordenados
            ith_cluster_silhouette_values = silhouette_vals[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / k)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, ith_cluster_silhouette_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # Espaço entre os clusters
        
        ax_sil.set_title("Gráfico de Silhueta para k = {}".format(k))
        ax_sil.set_xlabel("Coeficiente de Silhueta")
        ax_sil.set_ylabel("Clusters")
        ax_sil.axvline(x=sil_score, color="red", linestyle="--", label=f"Média: {sil_score:.2f}")
        ax_sil.legend(loc="upper right")
        ax_sil.set_yticks([])  # Remove marcas no eixo y
        ax_sil.set_xlim([-0.1, 1])
        st.pyplot(fig_sil)
        
        # Gráfico do Método do Cotovelo
        if st.checkbox("Mostrar Método do Cotovelo"):
            inertia_values = []
            k_range = range(2, 11)
            for n in k_range:
                kmeans_temp = KMeans(n_clusters=n, random_state=42, n_init=10)
                kmeans_temp.fit(data_scaled)
                inertia_values.append(kmeans_temp.inertia_)
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
            ax_elbow.plot(list(k_range), inertia_values, marker='o')
            ax_elbow.set_title("Método do Cotovelo")
            ax_elbow.set_xlabel("Número de Clusters")
            ax_elbow.set_ylabel("Inércia")
            st.pyplot(fig_elbow)
        
        # Visualização PCA com clusters
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = ax1.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='tab20', alpha=0.7)
        ax1.set_xlabel('Componente Principal 1')
        ax1.set_ylabel('Componente Principal 2')
        plt.colorbar(scatter, ax=ax1)
        st.pyplot(fig1)
        
        # Tabela de médias dos clusters
        st.subheader("Características dos Clusters")
        cluster_stats = data.groupby('Cluster')[selected_features].mean()
        cluster_stats = cluster_stats.rename(columns=FEATURE_DICT)
        st.write("**Médias Padronizadas por Cluster (Escala Z):**")
        styled_table = cluster_stats.style \
            .background_gradient(cmap='Blues', axis=0) \
            .format("{:.2f}") \
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#404040'), ('color', 'white')]
            }])
        st.dataframe(styled_table)
        st.write("A tabela exibe as médias dos valores padronizados (escala Z) para cada cluster.")
        
        # Análise Detalhada por Cluster
        st.subheader("Análise Detalhada por Cluster")
        boxplot_feature = st.selectbox(
            "Selecione a característica para visualização:",
            options=selected_features,
            format_func=lambda x: FEATURE_DICT[x],
            index=0
        )
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            x='Cluster',
            y=boxplot_feature,
            data=data,
            palette='Set2',
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
        )
        ax2.set_title(f'Distribuição de {FEATURE_DICT[boxplot_feature]} por Cluster')
        ax2.set_ylabel('Valor (USD)')
        st.pyplot(fig2)
        
        # Cidades por Cluster
        st.subheader("Cidades por Cluster")
        selected_cluster = st.selectbox("Selecione um cluster:", range(k))
        st.write(data[data['Cluster'] == selected_cluster][['city', 'country']])
        st.write("Agrupa cidades com base em similaridades de custo de vida, possibilitando análise detalhada por cluster.")
