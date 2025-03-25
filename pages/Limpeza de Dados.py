import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

# Configurações
plt.style.use('ggplot')
st.set_page_config(page_title="City Data Cleanser Pro", layout="wide")

@lru_cache(maxsize=None)
def load_data():
    """Carrega dados com cache e tratamento de erros"""
    try:
        df = pd.read_csv('data.csv')
        return df
    except Exception as e:
        st.error(f"Erro no carregamento: {str(e)}")
        return pd.DataFrame()

def analyze_missing(df):
    """Analisa dados faltantes com visualização"""
    if df.empty:
        return pd.Series(dtype=float)
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    return missing[missing > 0]

def clean_data(raw_df):
    """Pipeline completo de limpeza de dados"""
    try:
        df = raw_df.copy()
        
        # 1. Dados geográficos essenciais
        df = df.dropna(subset=['city', 'country'])
        if df.empty:
            return pd.DataFrame()
        
        # 2. Categorização de produtos/serviços
        food_cols = ['x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15',
                    'x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27']
        transport_cols = ['x28','x29','x30','x31','x32','x33']
        services_cols = ['x39','x40','x41','x42','x43']
        economic_cols = ['x36','x38','x48','x54']

        # 3. Preenchimento hierárquico
        def fill_hierarchical(col, groups=['country', 'region']):
            for group in groups:
                if group in df.columns:
                    df[col] = df.groupby(group)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
            df[col] = df[col].fillna(df[col].median())
            return df[col]

        # 4. Aplicação das estratégias
        # 4.1 Alimentos e bebidas
        for col in food_cols:
            if col in df.columns:
                df[col] = fill_hierarchical(col)

        # 4.2 Transporte
        for col in transport_cols:
            if col in df.columns:
                df[col] = fill_hierarchical(col)

        # 4.3 Serviços opcionais
        for col in services_cols:
            if col in df.columns:
                df[f'has_{col}'] = np.where(df[col].isna(), 0, 1)
                df[col] = np.where(
                    df[f'has_{col}'] == 1,
                    fill_hierarchical(col),
                    0
                )

        # 4.4 Dados econômicos
        try:
            df['eco_category'] = pd.qcut(df['x54'], 3, labels=['low','medium','high'])
        except:
            df['eco_category'] = 'medium'
            
        for col in economic_cols:
            if col in df.columns:
                df[col] = df.groupby(['country','eco_category'])[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # 5. Tratamento de outliers
        def treat_outliers(col):
            try:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 3*iqr
                
                country_upper = df.groupby('country')[col].transform(
                    lambda x: x.quantile(0.95)
                )
                df[col] = np.where(df[col] > upper, country_upper, df[col])
            except:
                pass
            return df[col]
        
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = treat_outliers(col)

        # 6. Controle de qualidade
        df['quality_score'] = df[['x48','x54','x36']].notnull().mean(axis=1)
        
        return df.drop(columns=['region','eco_category'], errors='ignore')
    
    except Exception as e:
        st.error(f"Erro na limpeza: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("🌍 Data Cleasing do Dataset")


    st.write("A estratégia utilizada para o preenchimento dos dados foi a categorização dos mesmo, separando-os e aplicando diferentes tratamentos de acordo com o grupo inserido. O tratamento de outliers foi utilizando IQR para limitar valores extremos e substituir por percentis dentro de cada país")
    raw_df = load_data()
    
    if not raw_df.empty:
        # Seção inicial
        st.header("🔍 Initial Analysis")
        col1, col2 = st.columns(2)
        col1.metric("Total de Cidades", len(raw_df))
        col2.metric("Colunas", len(raw_df.columns))
        
        # Dados faltantes originais
        st.subheader("📉 Original Missing Data")
        missing_original = analyze_missing(raw_df)
        
        if not missing_original.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            missing_original.plot.bar(ax=ax, color='darkred')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Missing Values (%)')
            st.pyplot(fig)
        else:
            st.success("✅ No missing data found!")
        
        # Processamento
        if st.button("🧼 Clean Data", type="primary"):
            with st.spinner('Cleaning... This may take a few minutes'):
                cleaned_df = clean_data(raw_df)
                
                if not cleaned_df.empty:
                    # Resultados
                    st.header("📊 Cleaning Results")
                    
                    # Métricas
                    cols = st.columns(3)
                    cols[0].metric("Valid Cities", 
                                f"{len(cleaned_df)} ({len(cleaned_df)/len(raw_df):.1%})")
                    
                    missing_cleaned = analyze_missing(cleaned_df)
                    # Correção do cálculo da redução de dados faltantes
                    reduction = (missing_original.sum() - missing_cleaned.sum()) / missing_original.sum() * 100
                    cols[1].metric("Missing Reduction", f"{reduction:.1f}%")
                    
                    cols[2].metric("Quality Score", 
                                f"{cleaned_df['quality_score'].mean():.2f}/1.0")
                    
                    # Comparação gráfica
                    st.subheader("🔄 Before/After Comparison")
                    fig, ax = plt.subplots(figsize=(12,6))
                    
                    width = 0.4
                    x = np.arange(len(missing_original))
                    
                    ax.bar(x - width/2, missing_original, width, label='Original', color='darkred')
                    ax.bar(x + width/2, missing_cleaned.reindex(missing_original.index, fill_value=0), 
                         width, label='Cleaned', color='lightgreen')
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(missing_original.index, rotation=45, ha='right')
                    ax.legend()
                    ax.set_ylabel('Missing Values (%)')
                    st.pyplot(fig)
                    
                    # Download
                    st.subheader("💾 Download Clean Data")
                    csv = cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="data_cleanedOF.csv",
                        mime="text/csv",
                        help="Download cleaned dataset in CSV format"
                    )
                else:
                    st.warning("No valid data after cleaning!")
    else:
        st.warning("⚠️ Please upload/place 'data.csv' in the directory!")

if __name__ == "__main__":
    main()
