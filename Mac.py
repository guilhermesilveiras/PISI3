import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Home",
)

# Carregar os dados
data = pd.read_csv("data.csv")

# Renomear colunas para facilitar o entendimento (opcional)
data.rename(columns={
    'city': 'Cidade',
    'country': 'País',
    'x3': 'McMeal (USD)',
    'x5': 'Cerveja Importada (USD)',
    'x54': 'Média Salário (USD)',
}, inplace=True)

# Título da aplicação
st.title("Análise Comparativa de Preços do McDonald's e Salários por País")


# Verificar se as colunas relevantes existem
if "McMeal (USD)" in data.columns and "Média Salário (USD)" in data.columns:
    # Agrupar por país e calcular o preço médio do McMeal e a média salarial
    agrupado_pais = data.groupby("País")[["McMeal (USD)", "Média Salário (USD)"]].mean().reset_index()
    
    # Ordenar os salários em ordem crescente
    agrupado_pais = agrupado_pais.sort_values(by="Média Salário (USD)", ascending=True)

    # Criando subplots para exibir ambos os gráficos lado a lado com maior espaçamento
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Preço Médio do McDonald's (McMeal) por País", "Média de Salário por País"),
        column_widths=[0.45, 0.45],  # Ajustando a largura das colunas
        horizontal_spacing=0.3  # Aumentando o espaçamento horizontal entre os gráficos
    )

    # Gráfico de barras horizontal para Preço Médio do McDonald's
    fig.add_trace(
        px.bar(
            agrupado_pais,
            x="McMeal (USD)",
            y="País",
            labels={"País": "País", "McMeal (USD)": "Preço Médio (USD)"},
            color="McMeal (USD)",
            color_continuous_scale="Viridis"
        ).data[0], row=1, col=1
    )

    # Gráfico de barras horizontal para Média de Salário
    fig.add_trace(
        px.bar(
            agrupado_pais,
            x="Média Salário (USD)",
            y="País",
            labels={"País": "País", "Média Salário (USD)": "Média Salário (USD)"},
            color="Média Salário (USD)",
            color_continuous_scale="Blues"
        ).data[0], row=1, col=2
    )

    # Atualizar layout para tornar os gráficos mais legíveis
    fig.update_layout(
        title_text="",
        showlegend=False,
        height=700,  # Aumentando a altura para um espaçamento maior
        width=1400,  # Aumentando a largura para espaçar os gráficos
        title_x=0.5,
        xaxis_title="Preço do McDonald's (USD)",
        yaxis_title="País",
        xaxis2_title="Média Salário (USD)",
        yaxis2_title="País",
        margin=dict(t=80, b=80, l=80, r=80)  # Ajustando margens (superior, inferior, esquerda e direita)
    )

    st.plotly_chart(fig)

    # Estatísticas descritivas por país (para McMeal e Salário)
    st.write("### Estatísticas por País (McDonald's e Salário)")
    estatisticas = agrupado_pais[["McMeal (USD)", "Média Salário (USD)"]].describe()
    st.write(estatisticas)
else:
    st.write("As colunas 'McMeal (USD)' ou 'Média Salário (USD)' não foram encontradas nos dados.")

# Rodapé
st.write("Desenvolvido com Streamlit e Plotly")
