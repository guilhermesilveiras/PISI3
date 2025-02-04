import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Carregar os dados
data = pd.read_csv('data.csv')

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
ax.set_title('Método do Cotovelo para Escolha do Número de Clusters', fontsize=14)
ax.set_xlabel('Número de Clusters', fontsize=12)
ax.set_ylabel('Inércia (Distortion)', fontsize=12)
ax.grid(True, linestyle='--', alpha=1)
ax.set_xticks(range_clusters)
ax.set_facecolor('none')
fig.patch.set_alpha(0)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

st.write("""
Após a análise, é possível perceber que o número de clusters ideal para o dataframe é 2. Vamos visualizar alguns gráfico resultado da clusterização:
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
kmeans = KMeans(n_clusters=2, random_state=42)  # Exemplo com 2 clusters
selected_data['cluster'] = kmeans.fit_predict(selected_data[['x48', 'x49']])

# Contar quantos dados caem em cada cluster por país
cluster_count = selected_data.groupby(['country', 'cluster']).size().reset_index(name='count')

# Criar o gráfico de dispersão
fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
sns.scatterplot(data=selected_data, x='country', y='x48', hue='cluster', palette='Set1', s=100, marker='o', edgecolor='white', ax=ax)
ax.set_title('Clusterização de Apartamentos na América do Sul (x48 e x49)')
ax.set_xlabel('País')
ax.set_ylabel('Frequência')
plt.xticks(rotation=90)

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
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Exemplo com 3 clusters
selected_data['cluster'] = kmeans.fit_predict(selected_data[['x48', 'x49']])

# Contar quantos dados caem em cada cluster por país
cluster_count = selected_data.groupby(['country', 'cluster']).size().reset_index(name='count')

# Criar o gráfico de dispersão
fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
sns.scatterplot(data=selected_data, x='country', y='x48', hue='cluster', palette='Set1', s=100, marker='o', edgecolor='white', ax=ax)
ax.set_title('Clusterização de Apartamentos na América do Norte (x48 e x49)')
ax.set_xlabel('País')
ax.set_ylabel('Preço de Apartamento no Centro (x48)')
plt.xticks(rotation=90)
ax.set_facecolor('none')
fig.patch.set_alpha(0)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

# Exibir a quantidade de dados por cluster para cada país
st.write(cluster_count)



# criando o modelo K-means com 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(pca_df[['PC1', 'PC2']])

# Adicionando os clusters ao DataFrame
pca_df['Cluster'] = kmeans.labels_

# Plotando os clusters
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=pca_df, s=100, alpha=0.7, edgecolor='black', ax=ax)
ax.set_title('Visualização dos Clusters com PCA', fontsize=14)
ax.set_xlabel('Componente Principal 1 (PC1)', fontsize=12)
ax.set_ylabel('Componente Principal 2 (PC2)', fontsize=12)
ax.legend(title='Clusters')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('none')
fig.patch.set_alpha(0)

st.pyplot(fig)