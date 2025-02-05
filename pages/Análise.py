# Import libraries
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
import streamlit as st

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
data_bad = data[data["data_quality"] == 0]

st.write("Com uma classificação que o próprio dataset traz se uma cidade possui 'dados bons' ou 'dados ruins', nota-se que aproximadamente 20% das cidades possui dados bons, ou seja informações quase completas de todas as colunas presentes")
# Mostrar as formas das divisões
st.write("Dados bons:", data_good.shape), st.write(
    "Dados ruins:", data_bad.shape)


# apenas 20% dos dados sao bons(18,7)
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
fig, (axB, ax) = plt.subplots(nrows=2, figsize=(
    12, 10))  # Ajuste de tamanho do gráfico

# Gráfico para o número de cidades em cada país com base nos dados ruins
data_bad["country"].value_counts()[0:15].plot(
    kind='bar', color="blue", ax=axB)  # Alterei para 'bar' (barras verticais)
axB.set_title("Número de cidades por cada país com base nos dados ruins",
              color='black')  # Cor do título em preto
axB.tick_params(axis='x', colors='black')  # Cor dos rótulos do eixo X em preto
axB.tick_params(axis='y', colors='black')  # Cor dos rótulos do eixo Y em preto
axB.set_facecolor('none')  # Cor de fundo do gráfico transparente
fig.patch.set_alpha(0)  # Definir o fundo do gráfico como transparente

# Gráfico para o número de cidades em cada país com base nos dados bons
data_good["country"].value_counts()[0:15].plot(
    kind='bar', color="blue", ax=ax)  # Alterei para 'bar' (barras verticais)
ax.set_title("Número de cidades por cada país com base nos dados bons",
             color='black')  # Cor do título em preto
ax.tick_params(axis='x', colors='black')  # Cor dos rótulos do eixo X em preto
ax.tick_params(axis='y', colors='black')  # Cor dos rótulos do eixo Y em preto
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
    fig2, ax2 = plt.subplots(
        figsize=(11, 9), facecolor='none')  # Fundo transparente
    msno.matrix(data_good.iloc[:, 2:25], color=(0, 0, 0.8), ax=ax2)  # Gráfico
    ax2.set_title("Missing values (first variables)",
                  fontsize=12)  # Reduzir o tamanho do título
    # Reduzir tamanho das legendas no eixo x
    ax2.tick_params(axis='x', labelsize=8)
    # Reduzir tamanho das legendas no eixo y
    ax2.tick_params(axis='y', labelsize=8)
    fig2.patch.set_alpha(0)  # Transparência no fundo da figura
    st.pyplot(fig2)  # Exibir no Streamlit

    # Gráfico 2: Variáveis finais dos dados bons
    fig3, ax3 = plt.subplots(
        figsize=(11, 9), facecolor='none')  # Fundo transparente
    msno.matrix(data_good.iloc[:, 26:], color=(0, 0, 0.8), ax=ax3)  # Gráfico
    ax3.set_title("Missing values (remaining variables)",
                  fontsize=12)  # Reduzir o tamanho do título
    # Reduzir tamanho das legendas no eixo x
    ax3.tick_params(axis='x', labelsize=8)
    # Reduzir tamanho das legendas no eixo y
    ax3.tick_params(axis='y', labelsize=8)
    fig3.patch.set_alpha(0)  # Transparência no fundo da figura
    st.pyplot(fig3)  # Exibir no Streamlit

# Gráficos dos dados ruins (bad_data)
with col2:
    st.write("### Bad Data")

    # Gráfico 1: Variáveis iniciais dos dados ruins
    fig4, ax4 = plt.subplots(
        figsize=(11, 9), facecolor='none')  # Fundo transparente
    msno.matrix(data_bad.iloc[:, 2:25], color=(0, 0, 0.8), ax=ax4)  # Gráfico
    ax4.set_title("Missing values (first variables)",
                  fontsize=12)  # Reduzir o tamanho do título
    # Reduzir tamanho das legendas no eixo x
    ax4.tick_params(axis='x', labelsize=8)
    # Reduzir tamanho das legendas no eixo y
    ax4.tick_params(axis='y', labelsize=8)
    fig4.patch.set_alpha(0)  # Transparência no fundo da figura
    st.pyplot(fig4)  # Exibir no Streamlit

    # Gráfico 2: Variáveis finais dos dados ruins
    fig5, ax5 = plt.subplots(
        figsize=(11, 9), facecolor='none')  # Fundo transparente
    msno.matrix(data_bad.iloc[:, 26:], color=(0, 0, 0.8), ax=ax5)  # Gráfico
    ax5.set_title("Missing values (remaining variables)",
                  fontsize=12)  # Reduzir o tamanho do título
    # Reduzir tamanho das legendas no eixo x
    ax5.tick_params(axis='x', labelsize=8)
    # Reduzir tamanho das legendas no eixo y
    ax5.tick_params(axis='y', labelsize=8)
    fig5.patch.set_alpha(0)  # Transparência no fundo da figura
    st.pyplot(fig5)  # Exibir no Streamlit


st.write("Podemos notar certos padrões e anomalidades:")
st.write("-Nos dados ruins, algumas das colunas com mais dados faltantes são percebidas nas colunas x28 e x29, respectivamente one way ticket e monthly pass, que são passagens de transporte padronizados e passes para uso ilimitados")
st.write("-Muitas cidades não contém um transporte público uniformizado e padronizado, por isso nao fornecem essas informação nos casos")
st.write("-Acerca da anomalia na coluna x40, sobre o preço do aluguel da quadra de tênis, o acesso a uma quadra de tênis em muitas cidades pode ser difícil devido a escassez de quadras, o que pode explicar a falta de informações sobre")
st.write("-Quanto as colunas x52 e x53, referentes ao metro qudrado fora e dentro da cidade, em muitos locais as áreas para comprar são desvalorizadas devido a localidades monótonas")
st.write("-na good data a variável x43 é o custo anual do IPS (Escola Primária Internacional) para uma criança em dólares americanos. Não se deve ficar surpreso com a falta desta estatística no subconjunto de dados bons, uma vez que a maioria das cidades não fornece estes serviços")

st.write("")

st.write("Os dados faltantes no dataset não comprometem a funcionalidade principal do aplicativo, pois o sistema foi projetado para lidar com ausências de informações de maneira robusta, utilizando técnicas de tratamento e filtragem de dados que garantem que as funcionalidades essenciais, como visualizações, cálculos e análises, sejam realizadas sem interferência.")
