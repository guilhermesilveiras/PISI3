import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Home",
)

# Título da página
# titulo 1
st.title("Análise comparativa do custo de vida ao redor do mundo")

# propósito do trabalho
st.header("PISI3 — Global Cost of Living")
st.write("""
         Este projeto tem como objetivo principal analisar e explorar os dados encontrados no dataset "Global Cost of Living" (consultar link abaixo), buscando compreender como o preço de produtos e serviços, bem como os salários médios, variam entre diferentes países e continentes. Através dessa análise, esperamos encontra fatores que possam, eventualmente, influenciar a tomada de decisão de turistas no processo de escolha de um possível destino de viagem.""")

st.markdown(
    """
    <p style='font-size:12px'>🌐 dataset; <a href='https://www.kaggle.com/datasets/mvieira101/global-cost-of-living'>https://www.kaggle.com/datasets/mvieira101/global-cost-of-living</a></p>
    """, 
    unsafe_allow_html=True
)

st.write("""
         A análise desses dados é um trabalho conjunto com o artigo produzido pela Equipe 10, chamado: "Planejamento Inteligente de Viagens: Gestão e Organização baseada em dados", onde o foco principal é detalhar o desenvolvimento de um aplicativo que possa auxiliar o planejamento de viagens, buscando entender como a limitação orçamentária pode influenciar a escolha de destinos turísticos. Para isso, torna-se fundamental o entendimento da exploração realizada aqui, que visa identificar padrões que possam vir, ou não, a impactar o usuário no momento do planejamento de sua viagem.
            """)

st.markdown(
    """
    <div style='border: 2px solid #000000; padding: 10px; border-radius: 5px; background-color: #000000;'>
        <strong>Alunos:</strong>
        <ul>
            <li>Diego Clebson</li>
            <li>Daniel Santana</li>
            <li>Hivison Santos</li>
            <li>Guilherme Salgueiro</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)