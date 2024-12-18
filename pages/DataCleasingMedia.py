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

# Comparação de dados faltantes
st.write("**Comparação de Dados Faltantes Antes e Depois da Limpeza**")
st.write(f"**Quantidade de Valores Faltantes no Dataset Original**: {data.isnull().sum().sum()}")
st.write(f"**Quantidade de Valores Faltantes Após Limpeza**: {data_filled.isnull().sum().sum()}")

# Salvar o dataset limpo
data_filled.to_csv("data_cleaned.csv", index=False)

# Função para adicionar números nas colunas
def add_column_numbers(ax, df):
    total_columns = df.shape[1]  # Número total de colunas
    for i in range(total_columns):
        ax.text(
            x=i, 
            y=-0.3,  # Posição ajustada abaixo das colunas
            s=str(i + 1),  # Número da coluna (1-indexed)
            fontsize=12,  # Tamanho maior para melhor visualização
            color="black", 
            ha="center", 
            va="center",
            weight="bold"
        )

# Visualização dos dados faltantes no Dataset Original
st.write("**Visualização dos Dados Faltantes no Dataset Original**")
fig1, ax1 = plt.subplots(figsize=(14, 8))  # Tamanho maior
msno.matrix(data, ax=ax1, color=(0.2, 0.4, 0.6))  # Azul elegante
add_column_numbers(ax1, data)  # Adicionar números
st.pyplot(fig1)

# Visualização dos dados faltantes no Dataset Limpo
st.write("**Visualização dos Dados Faltantes Após a Limpeza**")
fig2, ax2 = plt.subplots(figsize=(14, 8))  # Tamanho maior
msno.matrix(data_filled, ax=ax2, color=(0.2, 0.4, 0.6))  # Mesmo tom de azul
add_column_numbers(ax2, data_filled)  # Adicionar números
st.pyplot(fig2)

# Relatório final
st.write("**Estratégia de Preenchimento de Dados Ausentes:**")
st.write(
    "Foi utilizado o método de mediana IQR para preencher os valores faltantes nas variáveis numéricas e o método de moda para preencher os valores faltantes nas variáveis categóricas."
)

st.write("**Relatório de Comparação - Limpeza de Dados**")
st.write(f"**Diferença no Total de Valores Faltantes**: {data.isnull().sum().sum() - data_filled.isnull().sum().sum()}")
