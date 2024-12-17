import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import missingno as msno

# Carregar o dataset original
data = pd.read_csv("data.csv")

# Título no Streamlit
st.title("Limpeza de Dados")

# Mostrar o dataset original
st.write("**Dataset Original**")
st.write(data.head())

# Remover Linhas ou Colunas Completamente Vazias
data_no_empty_cols = data.dropna(axis=1, how='all')  # Remover colunas totalmente vazias
data_no_empty_rows = data_no_empty_cols.dropna(axis=0, how='all')  # Remover linhas totalmente vazias

# Preenchendo os valores faltantes com a média (para variáveis numéricas)
data_filled = data_no_empty_rows.copy()

# Preencher variáveis numéricas com a média
for col in data_filled.select_dtypes(include=[np.number]).columns:
    data_filled[col].fillna(data_filled[col].mean(), inplace=True)

# Preencher variáveis categóricas com a moda (valor mais frequente)
for col in data_filled.select_dtypes(include=[object]).columns:
    data_filled[col].fillna(data_filled[col].mode()[0], inplace=True)

# Exibir o dataset limpo
st.write("**Dataset Após Limpeza**")
st.write(data_filled.head())

# Exibir a comparação entre dados antes e depois da limpeza
st.write("**Comparação de Dados Faltantes Antes e Depois da Limpeza**")
st.write(f"**Quantidade de Valores Faltantes no Dataset Original**: {data.isnull().sum().sum()}")
st.write(f"**Quantidade de Valores Faltantes Após Limpeza**: {data_filled.isnull().sum().sum()}")



data_filled.to_csv("data_cleaned.csv", index=False)

# Visualizar os gráficos de valores faltantes antes e depois da limpeza
st.write("**Visualização dos Dados Faltantes no Dataset Original**")
fig1, ax1 = plt.subplots(figsize=(12, 6))
msno.matrix(data, ax=ax1)
st.pyplot(fig1)

st.write("**Visualização dos Dados Faltantes Após a Limpeza**")
fig2, ax2 = plt.subplots(figsize=(12, 6))
msno.matrix(data_filled, ax=ax2)
st.pyplot(fig2)

st.write("Estratégia de Preenchimento de Dados Ausentes: Preenchemos os valores faltantes nesse processo de Data Cleansing utilizando uma abordagem simples, mas eficaz. Para as colunas numéricas, os valores ausentes foram substituídos pela média dos valores da coluna correspondente, garantindo que o impacto nos cálculos estatísticos futuros seja minimizado e que a distribuição dos dados permaneça consistente. Já para as colunas categóricas, utilizamos a moda (valor mais frequente), permitindo que categorias mais representativas preencham as lacunas de maneira lógica, preservando a coerência das informações categóricas. Estratégia para uma soluçãao mais simples e genérica, mantendo a coerência")


# Relatório de comparação
st.write("**Relatório de Comparação - Limpeza de Dados**")



# Exibir a diferença no total de dados faltantes
st.write(f"**Diferença no Total de Valores Faltantes**: {data.isnull().sum().sum() - data_filled.isnull().sum().sum()}")

st.write("**Estatísticas Descritivas do Dataset Original**")
st.write(data.describe())

st.write("**Estatísticas Descritivas do Dataset Limpo**")
st.write(data_filled.describe())