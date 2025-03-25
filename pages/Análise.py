# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno
from scipy import stats
import warnings

tab1, tab2, tab3, tab4 = st.tabs(["Análise de Qualidade dos Dados", "Índice BigMac", "Análise do custo de Transporte", "Análise do custo de Moradia"])

with tab1:

    # Ignorar avisos
    warnings.filterwarnings("ignore")

    # Definir o contexto e estilo do seaborn
    sns.set_context('notebook')
    sns.set_style("ticks")

    # Alterar o estilo do matplotlib para fundo escuro
    plt.style.use('dark_background')

    # Carregar o dataset
    data = pd.read_csv("data.csv")

    st.title("Descrição estatística e qualidade dos dados do dataset")
    # Exibir a descrição estatística do DataFrame
    st.write("Média de valores por coluna:     (count: quantidade de células não nulas / mean: média dos valores / std: desvio padrão)", data.describe()) 

    st.write("")
    st.write("")

    # Dividir os dados em 'bons' e 'ruins' com base na qualidade dos dados
    data_good = data[data["data_quality"] == 1]
    data_bad  = data[data["data_quality"] == 0]

    st.write("Com uma classificação que o próprio dataset traz se uma cidade possui 'dados bons' ou 'dados ruins', nota-se que aproximadamente 20% das cidades possui dados bons, ou seja informações quase completas de todas as colunas presentes")
    # Mostrar as formas das divisões
    st.write("Dados bons:", data_good.shape), st.write("Dados ruins:", data_bad.shape)


    #apenas 20% dos dados sao bons(18,7)
    # Remover a coluna 'data_quality' dos dois DataFrames
    data_good.drop("data_quality", inplace=True, axis=1) 
    data_bad.drop("data_quality", inplace=True, axis=1)  

    # Exibir as primeiras linhas dos dados bons
    st.write("Primeiras linhas dos dados bons:")
    st.write(data_good.head())

    st.write("Primeiras linhas dos dados ruins:")
    st.write(data_bad.head())

    st.header("Cidades por país presentes")
    st.write("""
    Antes de partir para uma análise mais detalhada dos dados, é importante verificar que tipo de informação as cidades presentes no dataset podem nos trazer, sendo divididas por cidades com dados bons e dados ruins. Para construir essa visualização, vamos construir uma visão que possa mostrar quantas cidades existem em cada país em nosso dataset;
    """)


    # Criar subgráficos empilhados verticalmente (nrows=2 para duas linhas)
    fig, (axB, ax) = plt.subplots(nrows=2, figsize=(12, 10))  # Ajuste de tamanho do gráfico

    # Gráfico para o número de cidades em cada país com base nos dados ruins
    data_bad["country"].value_counts()[0:15].plot(kind='bar', color="blue", ax=axB)  # Alterei para 'bar' (barras verticais)
    axB.set_title("Número de cidades por cada país com base nos dados ruins", color= "gray")  # Alterado para cor automática
    axB.tick_params(axis='x', color= "gray")
    axB.tick_params(axis='y', color= "gray")
    axB.set_facecolor('none')  # Cor de fundo do gráfico transparente
    fig.patch.set_alpha(0)  # Definir o fundo do gráfico como transparente

    # Gráfico para o número de cidades em cada país com base nos dados bons
    data_good["country"].value_counts()[0:15].plot(kind='bar', color="blue", ax=ax)  # Alterei para 'bar' (barras verticais)
    ax.set_title("Número de cidades por cada país com base nos dados bons", color = "gray") 
    ax.tick_params(axis='x', color= 'gray')
    ax.tick_params(axis='y', color= "gray")
    ax.set_facecolor('none')  # Cor de fundo do gráfico transparente
    fig.patch.set_alpha(0)  # Definir o fundo do gráfico como transparente

    plt.subplots_adjust(hspace=0.4)  # Ajuste o espaço entre os gráficos verticais
    # Exibe os gráficos um abaixo do outro
    st.pyplot(fig)

    # Criar a estrutura de colunas para os gráficos
    st.header("Visão geral dos dados faltantes divididos por cidades que possuem dados bons e dados ruins")
    # Criar a estrutura de colunas para os gráficos
    col1, col2 = st.columns(2)  # Duas colunas para exibir os gráficos lado a lado

    # Gráficos dos dados bons (good_data)
    with col1:
        st.write("### Good Data")
        
        # Gráfico 1: Variáveis iniciais dos dados bons
        fig2, ax2 = plt.subplots(figsize=(11, 9), facecolor='none')  # Fundo transparente
        msno.matrix(data_good.iloc[:, 2:25], color=(0, 0, 0.8), ax=ax2)  # Gráfico
        ax2.set_title("Missing values (first variables)", fontsize=12)  # Reduzir o tamanho do título
        ax2.tick_params(axis='x', labelsize=8)  # Reduzir tamanho das legendas no eixo x
        ax2.tick_params(axis='y', labelsize=8)  # Reduzir tamanho das legendas no eixo y
        fig2.patch.set_alpha(0)  # Transparência no fundo da figura
        st.pyplot(fig2)  # Exibir no Streamlit

        # Gráfico 2: Variáveis finais dos dados bons
        fig3, ax3 = plt.subplots(figsize=(11, 9), facecolor='none')  # Fundo transparente
        msno.matrix(data_good.iloc[:, 26:], color=(0, 0, 0.8), ax=ax3)  # Gráfico
        ax3.set_title("Missing values (remaining variables)", fontsize=12)  # Reduzir o tamanho do título
        ax3.tick_params(axis='x', labelsize=8)  # Reduzir tamanho das legendas no eixo x
        ax3.tick_params(axis='y', labelsize=8)  # Reduzir tamanho das legendas no eixo y
        fig3.patch.set_alpha(0)  # Transparência no fundo da figura
        st.pyplot(fig3)  # Exibir no Streamlit

    # Gráficos dos dados ruins (bad_data)
    with col2:
        st.write("### Bad Data")
        
        # Gráfico 1: Variáveis iniciais dos dados ruins
        fig4, ax4 = plt.subplots(figsize=(11, 9), facecolor='none')  # Fundo transparente
        msno.matrix(data_bad.iloc[:, 2:25], color=(0, 0, 0.8), ax=ax4)  # Gráfico
        ax4.set_title("Missing values (first variables)", fontsize=12)  # Reduzir o tamanho do título
        ax4.tick_params(axis='x', labelsize=8)  # Reduzir tamanho das legendas no eixo x
        ax4.tick_params(axis='y', labelsize=8)  # Reduzir tamanho das legendas no eixo y
        fig4.patch.set_alpha(0)  # Transparência no fundo da figura
        st.pyplot(fig4)  # Exibir no Streamlit

        # Gráfico 2: Variáveis finais dos dados ruins
        fig5, ax5 = plt.subplots(figsize=(11, 9), facecolor='none')  # Fundo transparente
        msno.matrix(data_bad.iloc[:, 26:], color=(0, 0, 0.8), ax=ax5)  # Gráfico
        ax5.set_title("Missing values (remaining variables)", fontsize=12)  # Reduzir o tamanho do título
        ax5.tick_params(axis='x', labelsize=8)  # Reduzir tamanho das legendas no eixo x
        ax5.tick_params(axis='y', labelsize=8)  # Reduzir tamanho das legendas no eixo y
        fig5.patch.set_alpha(0)  # Transparência no fundo da figura
        st.pyplot(fig5)  # Exibir no Streamlit


    st.write("Podemos notar certos padrões e anomalias:")
    st.write("- Nos dados ruins, algumas das colunas com mais dados faltantes são percebidas nas colunas **x28** e **x29**, respectivamente *one way ticket* e *monthly pass*, que são passagens de transporte padronizadas e passes para uso ilimitado.")
    st.write("- Muitas cidades não contêm um transporte público uniformizado e padronizado, por isso não fornecem essas informações.")
    st.write("- Acerca da anomalia na coluna **x40**, sobre o preço do aluguel da quadra de tênis, o acesso a uma quadra de tênis em muitas cidades pode ser difícil devido à escassez de quadras, o que pode explicar a falta de informações.")
    st.write("- Quanto às colunas **x52** e **x53**, referentes ao metro quadrado fora e dentro da cidade, em muitos locais as áreas para comprar são desvalorizadas devido a localidades monótonas.")
    st.write("- Na *good data*, a variável **x43** é o custo anual do *IPS* (Escola Primária Internacional) para uma criança em dólares americanos. Não se deve ficar surpreso com a falta desta estatística no subconjunto de dados bons, uma vez que a maioria das cidades não fornece estes serviços.")

    st.write("")

    st.write("Os dados faltantes no dataset não comprometem a funcionalidade principal do aplicativo, pois o sistema foi projetado para lidar com ausências de informações de maneira robusta, utilizando técnicas de tratamento e filtragem de dados que garantem que as funcionalidades essenciais, como visualizações, cálculos e análises, sejam realizadas sem interferência.")

with tab2: 
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
                color_discrete_sequence=["blue"]  # Cor sólida azul
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
                color_discrete_sequence=["blue"]  # Cor sólida azul
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

with tab3:
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
    st.sidebar.header("Filtros para o Transporte")
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

with tab4:
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
    }, inplace=True)

    # Adicionar coluna de continente
    data['Continente'] = data['País'].apply(determinar_continente)

    # Remover dados com Continente desconhecido
    data = data[data['Continente'] != "Desconhecido"]

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