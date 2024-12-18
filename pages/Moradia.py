import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit as st

# Carregar o dataset
data = pd.read_csv("data_cleaned.csv")

# Lista de países e seus respectivos continentes
continentes = {
    'Africa': ['Algeria', 'Angola', 'Botswana', 'Egypt', 'South Africa', 'Morocco', 'Nigeria', 'Kenya', 'Tunisia', 'Ethiopia'],
    'Asia': ['China', 'Japan', 'India', 'South Korea', 'Singapore', 'Saudi Arabia', 'Turkey', 'Indonesia', 'Israel', 'Vietnam'],
    'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Russia', 'Netherlands', 'Switzerland', 'Sweden', 'Poland'],
    'North America': ['United States', 'Canada', 'Mexico', 'Cuba', 'Dominican Republic'],
    'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela'],
    'Oceania': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji']
}

# Função para determinar o continente com base no país
def determinar_continente(pais):
    for continente, paises in continentes.items():
        if pais in paises:
            return continente
    return "Desconhecido"
# Carregar o dataset
data = pd.read_csv("data_cleaned.csv")

# Lista de países e seus respectivos continentes
continentes = {
    'Africa': ['Algeria', 'Angola', 'Botswana', 'Egypt', 'South Africa', 'Morocco', 'Nigeria', 'Kenya', 'Tunisia', 'Ethiopia'],
    'Asia': ['China', 'Japan', 'India', 'South Korea', 'Singapore', 'Saudi Arabia', 'Turkey', 'Indonesia', 'Israel', 'Vietnam'],
    'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Russia', 'Netherlands', 'Switzerland', 'Sweden', 'Poland'],
    'North America': ['United States', 'Canada', 'Mexico', 'Cuba', 'Dominican Republic'],
    'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela'],
    'Oceania': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji']
}

# Função para determinar o continente com base no país
def determinar_continente(pais):
    for continente, paises in continentes.items():
        if pais in paises:
            return continente
    return "Desconhecido"

# Renomear colunas para facilitar o entendimento
data.rename(columns={
    'city': 'Cidade',
    'country': 'País',
    'x48': 'Aluguel 1 Quarto no Centro (USD)'  # Aluguel de um quarto no centro
    'x48': 'Aluguel 1 Quarto no Centro (USD)'  # Aluguel de um quarto no centro
}, inplace=True)

# Adicionar coluna de continente
data['Continente'] = data['País'].apply(determinar_continente)
# Adicionar coluna de continente
data['Continente'] = data['País'].apply(determinar_continente)

# Remover dados com Continente desconhecido
data = data[data['Continente'] != "Desconhecido"]
# Remover dados com Continente desconhecido
data = data[data['Continente'] != "Desconhecido"]

# Calcular a média do aluguel por continente
media_aluguel_continente = data.groupby('Continente')['Aluguel 1 Quarto no Centro (USD)'].mean().reset_index()
# Calcular a média do aluguel por continente
media_aluguel_continente = data.groupby('Continente')['Aluguel 1 Quarto no Centro (USD)'].mean().reset_index()

# Criar gráfico de barras com a coloração baseada na tonalidade azul (Viridis)
fig = px.bar(
    media_aluguel_continente,
    x="Continente",
    y="Aluguel 1 Quarto no Centro (USD)",
    color="Aluguel 1 Quarto no Centro (USD)",
    title="Custo Médio de Aluguel de 1 Quarto no Centro por Continente",
    labels={"Aluguel 1 Quarto no Centro (USD)": "Custo do Aluguel (USD)", "Continente": "Continente"},
    color_continuous_scale='Viridis',  # Usando a paleta 'Viridis' para o gráfico de calor
    height=500
)

# Ajustar layout do gráfico
fig.update_layout(
    xaxis=dict(title="Continente"),
    yaxis=dict(title="Custo Médio do Aluguel (USD)"),
    plot_bgcolor='rgba(255,255,255,1)',  # Fundo branco para o modo claro
    paper_bgcolor='rgba(255,255,255,1)',  # Fundo branco para o modo claro
    font=dict(color='black')  # Cor do texto preta para contraste com fundo branco
)

# Exibir o gráfico no Streamlit
st.title("Análise de Custo de Vida por Continente")
st.plotly_chart(fig)

st.write("O gráfico apresenta o Custo Médio de Aluguel de 1 Quarto no Centro por continente, evidenciando as disparidades regionais nos preços. Observa-se que a América do Norte possui o custo mais elevado, ultrapassando 1.100 USD, seguida pela Oceania com valores próximos a 1.000 USD. A Europa também apresenta valores consideráveis, situando-se em torno de 700 USD. Em contrapartida, continentes como África, Ásia e América do Sul exibem custos significativamente menores, todos abaixo dos 500 USD, indicando um padrão de preços mais acessível nessas regiões. O gráfico utiliza uma paleta de cores gradiente, indo do roxo ao amarelo, para representar a variação nos valores, facilitando a comparação visual entre as regiões.")

# Exibir estatísticas descritivas para o custo de aluguel por continente
st.write("### Estatísticas Descritivas por Continente")
st.write(media_aluguel_continente.describe())