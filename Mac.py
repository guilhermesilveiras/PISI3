import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Home",
)

# Carregar os dados
data = pd.read_csv("data.csv")

# Renomear colunas para facilitar o entendimento
data.rename(columns={
    'city': 'Cidade',
    'country': 'País',
    'x3': 'McMeal (USD)',
    'x5': 'Cerveja Importada (USD)',
    'x54': 'Média Salário (USD)',
}, inplace=True)

# Criar um mapeamento de continentes para os países
continent_mapping = {
    'Asia': ['Japan', 'China', 'India', 'South Korea'],
    'Europe': ['Germany', 'France', 'Italy', 'Spain'],
    'Américas': ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile'],
    'Africa': ['South Africa', 'Nigeria', 'Egypt'],
    'Oceania': ['Australia', 'New Zealand']
}

# Adicionar uma coluna de continente ao dataset
def map_to_continent(country):
    for continent, countries in continent_mapping.items():
        if country in countries:
            return continent
    return 'Other'

data['Continente'] = data['País'].apply(map_to_continent)

# Título da aplicação
st.title("Análise Comparativa de Preços do McDonald's e Salários")

# Verificar se as colunas relevantes existem
if "McMeal (USD)" in data.columns and "Média Salário (USD)" in data.columns:
    # Filtrar países com dados disponíveis para ambas as colunas
    data_filtrada = data.dropna(subset=["McMeal (USD)", "Média Salário (USD)"])

    # **Análise por País**
    agrupado_pais = data_filtrada.groupby("País")[["McMeal (USD)", "Média Salário (USD)"]].mean().reset_index()
    agrupado_pais = agrupado_pais.sort_values(by="Média Salário (USD)", ascending=True)

    fig_pais = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Média de Salário", "Preço Médio do McDonald's (McMeal)"),
        column_widths=[0.45, 0.45],
        horizontal_spacing=0.3
    )

    # Gráfico de salário à esquerda
    fig_pais.add_trace(
        px.bar(
            agrupado_pais,
            x="Média Salário (USD)",
            y="País",
            orientation='h',
            color="Média Salário (USD)",
            color_continuous_scale="Blues",
            labels={"País": "País", "Média Salário (USD)": "Média Salário (USD)"}
        ).data[0], row=1, col=1
    )

    # Gráfico de McMeal à direita
    fig_pais.add_trace(
        px.bar(
            agrupado_pais,
            x="McMeal (USD)",
            y="País",
            orientation='h',
            color="McMeal (USD)",
            color_continuous_scale="Viridis",
            labels={"País": "País", "McMeal (USD)": "Preço Médio (USD)"}
        ).data[0], row=1, col=2
    )

    fig_pais.update_layout(
        title_text="Análise Geral por País",
        showlegend=False,
        height=700,
        width=1400,
        title_x=0
    )
    st.plotly_chart(fig_pais)

    # **Gráficos para Extremos**
    extremos_pais = pd.concat([agrupado_pais.head(25), agrupado_pais.tail(25)])

    fig_extremos = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Média de Salário", "Preço Médio do McDonald's (McMeal)"),
        column_widths=[0.45, 0.45],
        horizontal_spacing=0.3
    )

    # Gráfico de salário à esquerda
    fig_extremos.add_trace(
        px.bar(
            extremos_pais,
            x="Média Salário (USD)",
            y="País",
            orientation='h',
            color="Média Salário (USD)",
            color_continuous_scale="Turbo",
            labels={"País": "País", "Média Salário (USD)": "Média Salarial (USD)"}
        ).data[0], row=1, col=1
    )

    # Gráfico de McMeal à direita
    fig_extremos.add_trace(
        px.bar(
            extremos_pais,
            x="McMeal (USD)",
            y="País",
            orientation='h',
            color="McMeal (USD)",
            color_continuous_scale="Viridis",
            labels={"País": "País", "McMeal (USD)": "Preço Médio McMeal (USD)"}
        ).data[0], row=1, col=2
    )

    fig_extremos.update_layout(
        title_text="Análise dos 25 maiores/menores médias de salário por País (USD)",
        showlegend=False,
        height=700,
        width=1400,
        title_x=0
    )
    st.plotly_chart(fig_extremos)

    # **Análise por Continente**
    agrupado_continente = data_filtrada.groupby("Continente")[["McMeal (USD)", "Média Salário (USD)"]].mean().reset_index()
    agrupado_continente = agrupado_continente.sort_values(by="Média Salário (USD)", ascending=True)

    fig_continente = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Média de Salário", "Preço Médio do McDonald's (McMeal)"),
        column_widths=[0.45, 0.45],
        horizontal_spacing=0.3
    )

    # Gráfico de salário à esquerda
    fig_continente.add_trace(
        px.bar(
            agrupado_continente,
            x="Média Salário (USD)",
            y="Continente",
            orientation='h',
            color="Média Salário (USD)",
            color_continuous_scale="Blues",
            labels={"Continente": "Continente", "Média Salário (USD)": "Média Salarial (USD)"}
        ).data[0], row=1, col=1
    )

    # Gráfico de McMeal à direita
    fig_continente.add_trace(
        px.bar(
            agrupado_continente,
            x="McMeal (USD)",
            y="Continente",
            orientation='h',
            color="McMeal (USD)",
            color_continuous_scale="Viridis",
            labels={"Continente": "Continente", "McMeal (USD)": "Preço Médio McMeal (USD)"}
        ).data[0], row=1, col=2
    )

    fig_continente.update_layout(
        title_text="Comparação de Salário Médio e Preço do McMeal por Continente (USD)",
        showlegend=False,
        height=600,
        width=1000,
        title_x=0
    )
    st.plotly_chart(fig_continente)

    # **Estatísticas descritivas**
    st.write("### Estatísticas por País (Salário Médio e McDonald's)")
    st.write(agrupado_pais[["McMeal (USD)", "Média Salário (USD)"]].describe())

else:
    st.write("As colunas 'McMeal (USD)' ou 'Média Salário (USD)' não foram encontradas nos dados.")
