# Importar bibliotecas
import pandas as pd
import plotly.express as px
import streamlit as st

# Carregar o dataset
data = pd.read_csv("data.csv")

# Carregar a lista de países e seus respectivos continentes
continentes = {
    'Africa': [
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 
        'Comoros', 'Congo (Congo-Brazzaville)', 'Democratic Republic of the Congo (Congo-Kinshasa)', 'Djibouti', 'Egypt', 'Equatorial Guinea', 
        'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 
        'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 
        'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 
        'Sudan', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
    ],
    'Asia': [
        'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 
        'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 
        'Malaysia', 'Maldives', 'Mongolia', 'Myanmar (Burma)', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 
        'Qatar', 'Russia', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste', 
        'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'
    ],
    'Europe': [
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 
        'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 
        'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 
        'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 
        'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'
    ],
    'North America': [
        'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 
        'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 
        'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'
    ],
    'South America': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
    ],
    'Oceania': [
        'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 
        'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'
    ]
}

# Dividir os dados em "good data" e "bad data"
data_good = data[data["data_quality"] == 1]
data_bad  = data[data["data_quality"] == 0]

# Remover a coluna 'data_quality' dos dois DataFrames
data_good.drop("data_quality", inplace=True, axis=1) 
data_bad.drop("data_quality", inplace=True, axis=1)

# Função para determinar o continente com base no país
def determinar_continente(pais):
    for continente, paises in continentes.items():
        if pais in paises:
            return continente
    return "Desconhecido"  # Caso o país não esteja na lista, retorna 'Desconhecido'

# Adicionar coluna de continente para ambos os DataFrames
data_good['continente'] = data_good['country'].apply(determinar_continente)
data_bad['continente'] = data_bad['country'].apply(determinar_continente)

# Contar quantos "good data" e "bad data" existem por continente
good_data_count = data_good['continente'].value_counts()
bad_data_count = data_bad['continente'].value_counts()

# Criar gráfico interativo para "good data" por continente
fig_good = px.bar(good_data_count, 
                  x=good_data_count.index, 
                  y=good_data_count.values, 
                  color=good_data_count.index, 
                  title="Quantidade de Good Data por Continente",
                  labels={'x': 'Continente', 'y': 'Número de Good Data'},
                  color_discrete_sequence=['green'])

# Ajustar o fundo para transparente
fig_good.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)

# Criar gráfico interativo para "bad data" por continente
fig_bad = px.bar(bad_data_count, 
                 x=bad_data_count.index, 
                 y=bad_data_count.values, 
                 color=bad_data_count.index, 
                 title="Quantidade de Bad Data por Continente",
                 labels={'x': 'Continente', 'y': 'Número de Bad Data'},
                 color_discrete_sequence=['red'])

# Ajustar o fundo para transparente
fig_bad.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)

# Exibir os gráficos interativos
st.plotly_chart(fig_good)
st.plotly_chart(fig_bad)
