import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Método do cotovelo para determinar o número ideal de clusters
distortions = []  # Lista para armazenar a inércia
range_clusters = range(1, 11)  # Testar de 1 a 10 clusters

# Aplicando PCA para reduzir para 2 componentes principais
pca = PCA(n_components=2)
pca_components = pca.fit_transform(numeric_data)  # 'numeric_data' é o seu dataset com variáveis numéricas

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

# Escolher o número ideal de clusters com base no método do cotovelo
optimal_clusters = 3  # Substitua pelo número ideal identificado no gráfico

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(pca_df[['PC1', 'PC2']])  # Treinar o modelo
pca_df['Cluster'] = kmeans_labels  # Adicionar rótulos ao DataFrame

# Gráfico de dispersão com clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Cluster', 
    palette='viridis', 
    data=pca_df, 
    s=100, 
    alpha=0.8
)
plt.title('Clusterização com K-Means baseada no PCA', fontsize=16)
plt.xlabel('Componente Principal 1 (PC1)', fontsize=12)
plt.ylabel('Componente Principal 2 (PC2)', fontsize=12)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
plt.legend(title='Cluster', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
