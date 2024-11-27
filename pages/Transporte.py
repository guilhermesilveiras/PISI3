import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Carregar os dados
data = pd.read_csv("data.csv")

# Renomear colunas para facilitar o entendimento
data.rename(columns={
    'city': 'Cidade',
    'country': 'País',
    'x3': 'McMeal (USD)',
    'x5': 'Cerveja Importada (USD)',
    'x54': 'Média Salário (USD)',
    'x28': 'Passagem Local (USD)',
    'x48': 'Aluguel 1 Quarto (USD)',
    'x31': 'Preço do Km do Táxi (USD)',
    'x33': 'Preço da Gasolina (USD)'
}, inplace=True)

# Título
st.title("Análise e Precificação de Viagens")

# Filtro Dinâmico por País
st.sidebar.header("Filtros")
selected_country = st.sidebar.selectbox("Selecione um País", options=["Todos"] + data['País'].unique().tolist())

# Filtrar os dados por país
filtered_data = data.copy()
if selected_country != "Todos":
    filtered_data = filtered_data[filtered_data["País"] == selected_country]

# Agora, ao selecionar o país, mostramos as cidades do país selecionado
cities_in_country = filtered_data['Cidade'].unique().tolist()

# Filtro por cidade dentro do país
selected_city = st.sidebar.selectbox("Selecione uma Cidade", options=["Todas"] + cities_in_country)

# Filtrar os dados por cidade, se necessário
if selected_city != "Todas":
    filtered_data = filtered_data[filtered_data["Cidade"] == selected_city]

# Agrupar os dados por cidade (dentro do país selecionado) e calcular as médias
avg_prices_by_city = filtered_data.groupby("Cidade")[["Preço do Km do Táxi (USD)", "Preço da Gasolina (USD)"]].mean().reset_index()

# Criar subplots para exibir gráficos lado a lado
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Preço do Km do Táxi por Cidade", "Preço da Gasolina por Cidade"),
    horizontal_spacing=0.2  # Espaçamento horizontal
)

# Gráfico do Preço do Km do Táxi por País
st.write("### Preço do Km do Táxi por País")
taxi_price_by_country = filtered_data.groupby("País")["Preço do Km do Táxi (USD)"].mean().reset_index()
fig_taxi_price_country = px.bar(
    taxi_price_by_country,
    y="País",  # País no eixo Y
    x="Preço do Km do Táxi (USD)",  # Preço no eixo X
    color="País",
    title="Preço Médio do Km do Táxi por País",
    labels={"Preço do Km do Táxi (USD)": "Preço Médio (USD)", "País": "País"},
    height=500
)
st.plotly_chart(fig_taxi_price_country)

# Gráfico do Preço da Gasolina por País
st.write("### Preço da Gasolina por País")
gas_price_by_country = filtered_data.groupby("País")["Preço da Gasolina (USD)"].mean().reset_index()
fig_gas_price_country = px.bar(
    gas_price_by_country,
    y="País",  # País no eixo Y
    x="Preço da Gasolina (USD)",  # Preço no eixo X
    color="País",
    title="Preço Médio da Gasolina por País",
    labels={"Preço da Gasolina (USD)": "Preço Médio (USD)", "País": "País"},
    height=500
)
st.plotly_chart(fig_gas_price_country)

# Gráfico de dispersão para Preço do Km do Táxi
fig.add_trace(
    go.Scatter(
        x=avg_prices_by_city["Preço do Km do Táxi (USD)"],  # Preço no eixo X
        y=avg_prices_by_city["Cidade"],  # Cidade no eixo Y
        mode="markers",
        name="Km do Táxi",
        marker=dict(color="blue", size=10)
    ),
    row=1, col=1
)

# Gráfico de dispersão para Preço da Gasolina
fig.add_trace(
    go.Scatter(
        x=avg_prices_by_city["Preço da Gasolina (USD)"],  # Preço no eixo X
        y=avg_prices_by_city["Cidade"],  # Cidade no eixo Y
        mode="markers",
        name="Gasolina",
        marker=dict(color="orange", size=10)
    ),
    row=1, col=2
)

# Configurar layout dos gráficos
fig.update_layout(
    title_text=f"Comparação de Preços: Km do Táxi e Gasolina por Cidade em {selected_country}",
    height=500,
    width=1000,
    title_x=0,
    showlegend=False,
    xaxis_title="Preço Médio (USD)",
    yaxis_title="Cidade"
)

st.plotly_chart(fig)

# Estatísticas descritivas
st.write("### Estatísticas Descritivas: Preço do Km do Táxi e da Gasolina por Cidade")
stats = avg_prices_by_city.describe()
st.write(stats)

# Rodapé
st.write("Desenvolvido com ❤️ para análise de viagens!")
