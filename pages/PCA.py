from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Carregando o dataset
data = pd.read_csv('PISI3/data.csv')

numeric_data = data.select_dtypes(include=['float64', 'int64']).drop(columns=['data_quality'], errors='ignore')
# Preenchendo valores ausentes com a média
imputer = SimpleImputer(strategy="mean")
numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Normalizando os dados
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data_imputed)

# Aplicando PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(numeric_data_scaled)

# Criando um DataFrame com as duas primeiras componentes principais
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['city'] = data['city']
pca_df['country'] = data['country']

# Método do cotovelo para determinar o número ideal de clusters
distortions = []  # Lista para armazenar a inércia
range_clusters = range(1, 11)  # Testar de 1 a 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_df[['PC1', 'PC2']])  # Usar apenas as primeiras duas componentes principais
    distortions.append(kmeans.inertia_)  # Adicionar a inércia do modelo atual

# Plotar o gráfico do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, distortions, marker='o', linestyle='--', color='b')
plt.title('Método do Cotovelo para Escolha do Número de Clusters', fontsize=14)
plt.xlabel('Número de Clusters', fontsize=12)
plt.ylabel('Inércia (Distortion)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range_clusters)
plt.show()