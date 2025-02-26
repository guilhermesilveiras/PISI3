import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Carregar os dados
data = pd.read_csv("data_cleaned.csv")

# Renomear colunas para facilitar o entendimento
data.rename(columns={
    'city': 'Cidade',
    'country': 'País',
    'x5': 'Cerveja Importada (USD)',
    'x54': 'Média Salário (USD)',
    'x28': 'Passagem Local (USD)',
    'x48': 'Aluguel 1 Quarto (USD)',
    'x31': 'Preço do Km do Táxi (USD)',
    'x33': 'Preço da Gasolina (USD)'
}, inplace=True)

# Mapeamento de países para continentes (adicionei países conhecidos, você pode expandir conforme necessário)
continent_mapping = {
    'África': ['South Africa', 'Nigeria', 'Egypt'],
    'Américas': ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile'],
    'Ásia': ['Japan', 'China', 'India', 'South Korea'],
    'Europa': ['Germany', 'France', 'Italy', 'Spain', 'United Kingdom', 'Portugal', 'Netherlands', 'Poland', 'Russia'],
    'Oceania': ['Australia', 'New Zealand']
}

# Função para mapear os países aos continentes
def map_to_continent(country):
    for continent, countries in continent_mapping.items():
        if country in countries:
            return continent
    return 'Outro'  # Caso não encontre um continente para o país

# Adicionar coluna de Continente ao dataset
data['Continente'] = data['País'].apply(map_to_continent)

# Título
st.title("Análise e Precificação de Viagens")

# Filtros dependentes
st.sidebar.header("Filtros")
selected_continent = st.sidebar.selectbox("Selecione um Continente", options=["Todos"] + sorted(data['Continente'].unique()))

# Filtrar países com base no continente selecionado
if selected_continent != "Todos":
    countries_in_continent = sorted(data[data['Continente'] == selected_continent]['País'].unique())
else:
    countries_in_continent = sorted(data['País'].unique())  # Incluir todos os países
selected_country = st.sidebar.selectbox("Selecione um País", options=["Todos"] + countries_in_continent)

# Filtrar cidades com base no país selecionado
if selected_country != "Todos":
    cities_in_country = sorted(data[data['País'] == selected_country]['Cidade'].unique())
else:
    if selected_continent != "Todos":
        cities_in_country = sorted(data[data['Continente'] == selected_continent]['Cidade'].unique())
    else:
        cities_in_country = sorted(data['Cidade'].unique())  # Incluir todas as cidades
selected_city = st.sidebar.selectbox("Selecione uma Cidade", options=["Todas"] + cities_in_country)

# Aplicar filtros no dataset
if selected_city != "Todas":
    filtered_data = data[data['Cidade'] == selected_city]
elif selected_country != "Todos":
    filtered_data = data[data['País'] == selected_country]
elif selected_continent != "Todos":
    filtered_data = data[data['Continente'] == selected_continent]
else:
    filtered_data = data

# Ajustar exibição dos gráficos com base nos filtros aplicados
if selected_city != "Todas":
    title = f"Análise para a Cidade {selected_city}"
    grouping_col = "Cidade"
elif selected_country != "Todos":
    title = f"Análise para o País {selected_country}"
    grouping_col = "Cidade"
elif selected_continent != "Todos":
    title = f"Análise para o Continente {selected_continent}"
    grouping_col = "País"
else:
    title = "Análise por Continente"
    grouping_col = "Continente"

# Agrupar os dados para o gráfico
grouped_data = filtered_data.groupby(grouping_col)[["Preço do Km do Táxi (USD)", "Preço da Gasolina (USD)"]].mean().reset_index()

# Exibir gráficos separados
st.write(f"### {title}")

# Função para gerar gráficos com escala de cor quente
def plot_heatmap_chart(df, x_col, y_col, title, color_col, color_scale):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: "Preço (USD)"},
        height=400,
        color=color_col,
        color_continuous_scale=color_scale  # Aplicando a escala de cores personalizada
    )
    st.plotly_chart(fig)

# Gráfico: Preço do Km do Táxi (com escala de cores "Viridis" para maior contraste)
st.write("#### Preço do Km do Táxi")
plot_heatmap_chart(grouped_data, grouping_col, "Preço do Km do Táxi (USD)", "Preço Médio do Km do Táxi", "Preço do Km do Táxi (USD)", "Viridis")

# Gráfico: Preço da Gasolina (com escala de cores "Plasma" para maior contraste)
st.write("#### Preço da Gasolina")
plot_heatmap_chart(grouped_data, grouping_col, "Preço da Gasolina (USD)", "Preço Médio da Gasolina", "Preço da Gasolina (USD)", "Plasma")

# Estatísticas descritivas dos dados filtrados
st.write("### Estatísticas Descritivas dos Dados Filtrados")
stats = filtered_data.describe()
st.write(stats)

st.write("A relação entre o preço da gasolina e o preço do táxi pode fornecer insights sobre a economia local. Se o preço do táxi for significativamente mais alto do que o custo da gasolina, pode indicar que a infraestrutura de transporte é deficiente ou que o mercado de táxi é monopolizado. Essa informação pode ser útil para ajustar preços de serviços dentro do aplicativo, oferecendo melhores opções de acordo com a região. Além disso, o preço do combustível e do táxi pode estar correlacionado com o poder de compra local e o custo de vida, o que pode ajudar a planejar novas opções de transporte ou parcerias estratégicas.")
st.write("Uma análise perceptível que, por exemplo, na Europa, a gasolina é mais cara do que o táxi, o que pode ser um reflexo de políticas fiscais que buscam desincentivar o consumo de combustível e promover o uso de transporte público ou alternativas mais sustentáveis.")
st.write("Também obversava-se na Oceania, onde o preço do táxi é mais caro que o da gasolina, isso pode ser atribuído a fatores como a baixa densidade populacional e a falta de alternativas de transporte público eficientes. O custo elevado do táxi pode ser uma consequência da maior dependência de transporte particular em algumas áreas. ")