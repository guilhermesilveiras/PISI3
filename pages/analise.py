import streamlit as st
import pandas as pd 
import geopandas as gpd
import geopandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# Carregar os dados
data = pd.read_csv("data.csv")

# filtrar por qualidade de dados
data_good = data[data["data_quality"] == 1]
data_bad  = data[data["data_quality"] == 0]

data_bad.drop("data_quality", inplace = True, axis = 1) 
data_good.drop("data_quality", inplace = True, axis = 1)

st.header("Qualidade e análise de dados")
st.write("""
         Antes de partir para uma análise mais detalhada dos dados, é importante verificar que tipo de informação as cidades presentes no dataset podem nos trazer. Para construir essa visualização, vamos construir uma visão que possa mostrar quantas cidades existem em cada país em nosso dataset;""")

# Plotar o número de cidades em cada país a partir dos dados bons
fig, ax = plt.subplots()

# Obter os valores e os países
values = data_good["country"].value_counts()[0:15]
countries = values.index

# Criar um degradê de cores
colors = plt.cm.Blues(np.linspace(0.3, 1, len(values)))

# Plotar os dados
bars = ax.barh(countries, values, color=colors)

# Definir o fundo do gráfico como transparente
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Configurar os rótulos e o título
ax.set_xlabel('Número de Cidades', color='white')
ax.set_ylabel('País', color='white')
ax.set_title('Número de Cidades em cada País (Top 15)', color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Configurar a cor da borda do gráfico
for spine in ax.spines.values():
    spine.set_edgecolor('none')

