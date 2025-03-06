import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def comparison_page():
    st.header("🔍 Comparação Detalhada Entre Cidades")
    
    # Carregar dados
    @st.cache_data
    def load_comparison_data():
        try:
            df = pd.read_csv("data_cleaned.csv")
            df = df.dropna(subset=['city', 'country'])
            df['city_country'] = df['city'] + ', ' + df['country']  # Criar identificador único
            return df
        except FileNotFoundError:
            st.error("Arquivo não encontrado!")
            return None

    data = load_comparison_data()
    
    if data is not None:
        # Widgets de seleção
        col1, col2 = st.columns(2)
        with col1:
            city1 = st.selectbox("Selecione a primeira cidade:", 
                               options=data['city_country'].unique(),
                               index=0)
        with col2:
            city2 = st.selectbox("Selecione a segunda cidade:", 
                               options=data['city_country'].unique(),
                               index=1)
        
        # Selecionar características para comparação
        features = st.multiselect(
            "Selecione os itens para comparação:",
            options=[col for col in data.columns if col.startswith('x')],
            default=['x1', 'x48', 'x54', 'x28', 'x41']
        )
        
        if not features:
            st.warning("Selecione pelo menos um item para comparação!")
            return
        
        # Filtrar dados
        df_filter = data[data['city_country'].isin([city1, city2])]
        df_melted = df_filter.melt(id_vars=['city_country'], 
                                 value_vars=features,
                                 var_name='Item', 
                                 value_name='Preço (USD)')
        
        # Criar visualização
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plotar dados
        for i, city in enumerate(df_melted['city_country'].unique()):
            city_data = df_melted[df_melted['city_country'] == city]
            ax.bar([x + i*0.4 for x in range(len(features))], 
                   city_data['Preço (USD)'], 
                   width=0.4, 
                   label=city)
        
        # Configurar eixos
        ax.set_xticks([x + 0.2 for x in range(len(features))])
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylabel("Preço em USD")
        ax.set_title(f"Comparação: {city1} vs {city2}")
        ax.legend()
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Mostrar tabela comparativa
        st.subheader("Tabela Comparativa")
        pivot_table = df_filter.set_index('city_country')[features].T
        st.dataframe(pivot_table.style.highlight_max(color='#90EE90').highlight_min(color='#FFB6C1'))
        
        # Botão para download
        if st.button("Baixar Dados para Esta Comparação"):
            csv = pivot_table.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"comparacao_{city1}_vs_{city2}.csv",
                mime='text/csv'
            )


comparison_page()


