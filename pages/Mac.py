import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# Carregar os dados
data = pd.read_csv("data_cleaned.csv")

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

    # **Gráficos para Média entre Maiores/Menores**
    agrupado_pais = data_filtrada.groupby("País")[["McMeal (USD)", "Média Salário (USD)"]].mean().reset_index()
    agrupado_pais = agrupado_pais.sort_values(by="Média Salário (USD)", ascending=True)
    
    # Calcular a média dos 10 maiores e 10 menores
    maiores = agrupado_pais.head(50)
    menores = agrupado_pais.tail(50)
    
    media_maiores_salario = maiores["Média Salário (USD)"].mean()
    media_menores_salario = menores["Média Salário (USD)"].mean()
    media_maiores_mcmeal = maiores["McMeal (USD)"].mean()
    media_menores_mcmeal = menores["McMeal (USD)"].mean()

    # Criar um novo dataframe para exibir as médias
    medias = pd.DataFrame({
        'Categoria': ['50 Maiores', '50 Menores'],
        'Média Salário (USD)': [media_maiores_salario, media_menores_salario],
        'McMeal (USD)': [media_maiores_mcmeal, media_menores_mcmeal]
    })

    fig_extremos = make_subplots(
        rows=2, cols=1,  # Alterado para 2 linhas e 1 coluna
        subplot_titles=("Média de Salário", "Preço Médio do McDonald's (McMeal)"),
        vertical_spacing=0.3
    )

    # Gráfico de salário com cor sólida
    fig_extremos.add_trace(
        px.bar(
            medias,
            y="Média Salário (USD)",
            x="Categoria",
            orientation='v',  # Alterado para vertical
            labels={"Categoria": "Categoria", "Média Salário (USD)": "Média Salarial (USD)"},
            color_discrete_sequence=["blue"]  # Cor sólida vermelha
        ).data[0], row=1, col=1
    )

    # Gráfico de McMeal com cor sólida
    fig_extremos.add_trace(
        px.bar(
            medias,
            y="McMeal (USD)",
            x="Categoria",
            orientation='v',  # Alterado para vertical
            labels={"Categoria": "Categoria", "McMeal (USD)": "Preço Médio McMeal (USD)"},
            color_discrete_sequence=["blue"]  # Cor sólida vermelha
        ).data[0], row=2, col=1
    )

    fig_extremos.update_layout(
        title_text="Média entre os 50 maiores e 50 menores salários e preços do McMeal (USD)",
        showlegend=False,
        height=1000,  # Ajustado para acomodar 2 gráficos
        width=900,
        title_x=0
    )
    st.plotly_chart(fig_extremos)

    st.write("A comparação entre os dois gráficos mostra uma diferença notável entre os países com os maiores e os menores salários, tanto em termos de salário médio quanto de preço do McMeal. Nos países com os 50 maiores salários, a média salarial é significativamente mais alta (acima de 2.500 USD), enquanto nos países com os 50 menores salários, a média salarial fica abaixo de 500 USD. Curiosamente, o preço médio do McMeal segue uma tendência oposta: nos países com os menores salários, o preço do McMeal é mais baixo, abaixo de 3 USD, enquanto nos países com os maiores salários, o preço ultrapassa 8 USD. Isso sugere que, embora os países com salários mais altos tenham maior poder de compra, também apresentam um custo de vida mais elevado, refletido no preço dos produtos, como o McMeal.")

    # **Análise por Continente**
    agrupado_continente = data_filtrada.groupby("Continente")[["McMeal (USD)", "Média Salário (USD)"]].mean().reset_index()
    agrupado_continente = agrupado_continente.sort_values(by="Média Salário (USD)", ascending=True)

    fig_continente = make_subplots(
        rows=2, cols=1,  # Alterado para 2 linhas e 1 coluna
        subplot_titles=("Média de Salário", "Preço Médio do McDonald's (McMeal)"),
        vertical_spacing=0.3
    )

    # Gráfico de salário com cor sólida
    fig_continente.add_trace(
        px.bar(
            agrupado_continente,
            y="Média Salário (USD)",
            x="Continente",
            orientation='v',  # Alterado para vertical
            labels={"Continente": "Continente", "Média Salário (USD)": "Média Salarial (USD)"},
            color_discrete_sequence=["blue"]  # Cor sólida vermelha
        ).data[0], row=1, col=1
    )

    # Gráfico de McMeal com cor sólida
    fig_continente.add_trace(
        px.bar(
            agrupado_continente,
            y="McMeal (USD)",
            x="Continente",
            orientation='v',  # Alterado para vertical
            labels={"Continente": "Continente", "McMeal (USD)": "Preço Médio McMeal (USD)"},
            color_discrete_sequence=["blue"]  # Cor sólida vermelha
        ).data[0], row=2, col=1
    )

    fig_continente.update_layout(
        title_text="Comparação de Salário Médio e Preço do McMeal por Continente (USD)", 
        showlegend=False,
        height=1000,  # Ajustado para acomodar 2 gráficos
        width=900,
        title_x=0
    )
    st.plotly_chart(fig_continente)
    st.write("Os gráficos mostram uma relação entre o salário médio e o preço do McMeal nos diferentes continentes. Nos continentes com salários mais elevados, como Oceania e Europa, o preço do McMeal também tende a ser mais alto, ultrapassando 6 USD, sendo a Oceania a que apresenta os valores mais altos. Em contrapartida, nos continentes com salários mais baixos, como África e Ásia, o preço do McMeal é mais baixo, variando entre 3 e 4 USD. Isso indica que o preço do McMeal acompanha a média salarial, refletindo o poder de compra em cada região.")

    # **Estatísticas descritivas**
    st.write("### Estatísticas por País (Salário Médio e McDonald's)")
    st.write(agrupado_pais[["McMeal (USD)", "Média Salário (USD)"]].describe())

else:
    st.write("As colunas 'McMeal (USD)' ou 'Média Salário (USD)' não foram encontradas nos dados.")

agrupado_pais['Porcentagem McMeal do Salário'] = (agrupado_pais['McMeal (USD)'] / agrupado_pais['Média Salário (USD)']) * 100

# Exibir as primeiras linhas da tabela com a porcentagem
st.write("### Porcentagem do Salário Representada pelo McMeal por País")
st.write(agrupado_pais[['País', 'McMeal (USD)', 'Média Salário (USD)', 'Porcentagem McMeal do Salário']])

st.write("O McMeal é uma referência global comum em um mundo com tantas disparidades econômicas, o que torna essa estatística essencial no desenvolvimento do aplicativo. Ao comparar o custo do McMeal com o salário médio, conseguimos oferecer uma visão única e relevante sobre a acessibilidade de alimentos em diferentes países, ajudando os usuários a planejar suas viagens de forma mais informada e adaptada à realidade local. Essa métrica possibilita decisões financeiras mais conscientes, considerando a diversidade de custos ao redor do mundo.")