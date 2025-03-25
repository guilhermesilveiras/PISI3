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

tab1, tab2 = st.tabs(["Início da Clusterização", "Clusters"])

with tab1:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import seaborn as sns

    # Carregar os dados
    data = pd.read_csv('data_cleaned.csv')

    # Selecionando apenas dados numéricos
    numeric_data = data.select_dtypes(include=['float64', 'int64']).drop(columns=['data_quality'], errors='ignore')

    # Exibir texto explicativo
    st.write("""
    ## Início da Clusterização com K-Means

    Primeiro, é preciso reduzir a dimensionalidade do nosso dataset, que contém mais de 50 colunas. Para isso, vamos utilizara técnica de redução de dimensionalidade, usada para criar um novo conjunto com Componentes Principais (PCs). Esses componentes são combinações lineares das variáveis originais que capturam a maior parte da variabilidade dos dados. Em seguida, aplicaremos o método de K-means para agrupar os países em clusters com base nas suas características.
    """)

    # Preenchendo valores ausentes com a média
    imputer = SimpleImputer(strategy="mean")
    numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)


    # Normalizando os dados
    scaler = StandardScaler()
    numeric_data_scaled = scaler.fit_transform(numeric_data_imputed)


    # Exibir texto explicativo
    st.write("""
    ### Normalização dos Dados

    Os dados são então normalizados para garantir que todas as variáveis contribuam igualmente para a análise.
    """)

    st.markdown(
        """
        <p style='font-size:12px'>scaler = StandartScaler()
        numeric_data_scaled = scaler.fit.transfor(nueric_data_imputed)</p>
        """, 
        unsafe_allow_html=True
    )




    # Aplicando PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(numeric_data_scaled)

    # Criando um DataFrame com as duas primeiras componentes principais
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    pca_df['city'] = data['city']
    pca_df['country'] = data['country']

    # Método do cotovelo 
    distortions = []  # lista para armazenar a inércia
    range_clusters = range(1, 11)  # testar de 1 a 10 clusters

    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pca_df[['PC1', 'PC2']])  # usar apenas as primeiras duas componentes principais
        distortions.append(kmeans.inertia_)  # adicionar a inércia do modelo atual

    # Exibir texto explicativo
    st.write("""
    ### Método do Cotovelo 💪

    Utilizamos o método do cotovelo para determinar o número ideal de clusters. O gráfico abaixo mostra a inércia (distortion) para diferentes números de clusters.
    """)

    # plotar o gráfico do cotovelo
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    ax.plot(range_clusters, distortions, marker='o', linestyle='--', color='b')
    ax.set_title('Método do Cotovelo para Escolha do Número de Clusters', fontsize=14, color= 'gray')
    ax.set_xlabel('Número de Clusters', fontsize=12, color= 'gray')
    ax.set_ylabel('Inércia (Distortion)', fontsize=12, color = 'gray')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(range_clusters)
    ax.set_facecolor('none')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    fig.patch.set_alpha(0)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    Após a análise, é possível perceber que o número de clusters ideal para o dataframe é 3. Vamos visualizar alguns gráfico resultado da clusterização:
    """)

    # Definir lista de países da América do Sul
    south_american_countries = [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 
        'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
    ]

    # Filtrar dados para incluir apenas os países da América do Sul
    south_american_data = data[data['country'].isin(south_american_countries)]

    # Selecionar as colunas relevantes (preços dos apartamentos e país)
    selected_data = south_american_data[['country', 'x48', 'x49']]

    # Remover as linhas com valores ausentes
    selected_data = selected_data.dropna()

    # realizar a clusterização com KMeans (número de clusters a ser ajustado)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Exemplo com 2 clusters
    selected_data['cluster'] = kmeans.fit_predict(selected_data[['x48', 'x49']])

    # Contar quantos dados caem em cada cluster por país
    cluster_count = selected_data.groupby(['country', 'cluster']).size().reset_index(name='count')

    # Criar o gráfico de dispersão
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
    sns.scatterplot(data=selected_data, x='country', y='x48', hue='cluster', palette='Set1', s=100, marker='o', edgecolor='white', ax=ax)
    ax.set_title('Clusterização de Apartamentos na América do Sul (x48 e x49)', color='gray')
    ax.set_xlabel('País', color= 'gray')
    ax.set_ylabel('Frequência', color= 'gray')
    plt.xticks(rotation=90, color= 'gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    # Tornar o fundo dos eixos transparente
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)

    st.pyplot(fig)

    # Exibir a quantidade de dados por cluster para cada país
    st.write(cluster_count)

    # Definir lista de países da América do Norte
    north_american_countries = ['Canada', 'United States', 'Mexico']

    # Filtrar dados para incluir apenas os países da América do Norte
    north_american_data = data[data['country'].isin(north_american_countries)]

    # Selecionar as colunas relevantes (preços dos apartamentos e país)
    selected_data = north_american_data[['country', 'x48', 'x49']]

    # Remover as linhas com valores ausentes
    selected_data = selected_data.dropna()

    # Realizar a clusterização com KMeans (número de clusters a ser ajustado)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Exemplo com 3 clusters
    selected_data['cluster'] = kmeans.fit_predict(selected_data[['x48', 'x49']])

    # Contar quantos dados caem em cada cluster por país
    cluster_count = selected_data.groupby(['country', 'cluster']).size().reset_index(name='count')

    # Criar o gráfico de dispersão
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
    sns.scatterplot(data=selected_data, x='country', y='x48', hue='cluster', palette='Set1', s=100, marker='o', edgecolor='white', ax=ax)
    ax.set_title('Clusterização de Apartamentos na América do Norte (x48 e x49)', color = 'gray')
    ax.set_xlabel('País', color= 'gray')
    ax.set_ylabel('Preço de Apartamento no Centro (x48)', color= 'gray')
    plt.xticks(rotation=90)
    ax.set_facecolor('none')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    fig.patch.set_alpha(0)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

    # Exibir a quantidade de dados por cluster para cada país
    st.write(cluster_count)



    # criando o modelo K-means com 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pca_df[['PC1', 'PC2']])

    # Adicionando os clusters ao DataFrame
    pca_df['Cluster'] = kmeans.labels_

    # Plotando os clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=pca_df, s=100, alpha=0.7, edgecolor='black', ax=ax)
    ax.set_title('Visualização dos Clusters com PCA', fontsize=14, color= 'gray')
    ax.set_xlabel('Componente Principal 1', fontsize=12, color= 'gray')
    ax.set_ylabel('Componente Principal 2', fontsize=12, color= 'gray')
    ax.legend(title='Clusters')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    st.pyplot(fig)

with tab2:

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
