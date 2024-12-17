import streamlit as st
import pandas as pd
import plotly.express as px

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
    'x48': 'Aluguel 1 Quarto no Centro (USD)',  # Aluguel de um quarto no centro da cidade
    'x31': 'Preço do Km do Táxi (USD)',
    'x33': 'Preço da Gasolina (USD)'
}, inplace=True)

# Título
st.title("Análise de Custo de Vida: Aluguel de Apartamento por País")

# Filtro Dinâmico por País
st.sidebar.header("Filtros")
selected_country = st.sidebar.selectbox("Selecione um País", options=["Todos"] + data['País'].unique().tolist())

# Filtrar os dados por país
filtered_data = data.copy()
if selected_country != "Todos":
    filtered_data = filtered_data[filtered_data["País"] == selected_country]

# Agrupar os dados por país e calcular as médias do aluguel de um quarto no centro
avg_rent_by_country = filtered_data.groupby("País")["Aluguel 1 Quarto no Centro (USD)"].mean().reset_index()

# Criar gráfico de barras para exibir o custo do aluguel de um quarto no centro por país
fig_rent_country = px.bar(
    avg_rent_by_country,
    x="País",
    y="Aluguel 1 Quarto no Centro (USD)",
    color="País",
    title="Custo de Aluguel de 1 Quarto no Centro da Cidade por País",
    labels={"Aluguel 1 Quarto no Centro (USD)": "Custo do Aluguel (USD)", "País": "País"},
    height=500
)

# Exibir o gráfico
st.plotly_chart(fig_rent_country)

# Estatísticas descritivas para o custo do aluguel
st.write("### Estatísticas Descritivas: Custo de Aluguel de 1 Quarto no Centro por País")
stats_rent = avg_rent_by_country.describe()
st.write(stats_rent)

st.write("O gráfica de média de aluguel oferece uma visão ampla sobre as variações nos preços em diferentes regiões. Esse tipo de análise é crucial para compreender as diferenças no custo de vida entre diversos destinos. Ao observar o gráfico, é possível notar que países com grandes centros urbanos e economias mais desenvolvidas, como Estados Unidos, Reino Unido ou Japão, apresentam aluguéis significativamente mais caros. Em contrapartida, países com menor custo de vida ou menos urbanos tendem a oferecer preços de aluguel mais acessíveis.")
st.write("Os países que se destacam pelos maiores preços, como Mônaco, Bermudas, Papua-Nova Guiné e Singapura possuem os maiores valores de aluguel devido à combinação de fatores como escassez de território, alta demanda por moradias e economias desenvolvidas. Em Mônaco e Singapura, a falta de espaço e a alta densidade populacional elevam os preços. Bermudas, com seu foco turístico e financeiro, atrai profissionais e visitantes de alto padrão, enquanto em Papua-Nova Guiné, a oferta restrita de imóveis adequados para expatriados e empresas multinacionais gera valores elevados. Esses fatores refletem a exclusividade e a competitividade do mercado imobiliário nessas regiões.")

