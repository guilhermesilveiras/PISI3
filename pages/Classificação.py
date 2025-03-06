import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial
st.set_page_config(page_title="Classificador de Custos de Viagem", layout="wide")

def criar_labels_reais(df):
    """Cria classificação real com distribuição balanceada"""
    # Limiares adaptativos baseados nos dados
    p_aluguel = df['x48'].quantile([0.85, 0.95]).values
    p_salario = df['x54'].quantile([0.80, 0.90]).values
    p_lazer = (df['x40'] + df['x41']).quantile([0.75, 0.90]).values
    
    # Condições melhoradas
    conditions = [
        # Premium Travel (Top 15% aluguel + Top 20% salário)
        (df['x48'] >= p_aluguel[0]) & (df['x54'] >= p_salario[0]),
        
        # City Explorer Luxury (Top 25% lazer + internet razoável)
        ((df['x40'] + df['x41']) >= p_lazer[0]) & (df['x38'] < df['x38'].median()),
        
        # Backpacker Budget (Baixo custo essencial)
        (df['x1'] <= df['x1'].quantile(0.25)) & 
        (df['x28'] <= df['x28'].quantile(0.25)),
    ]
    
    choices = [
        'Premium Travel',
        'City Explorer Luxury',
        'Backpacker Budget'
    ]
    
    df['real_class'] = np.select(
        conditions,
        choices,
        default='Mid-range Nomad'
    ).astype(str)
    
    # Garantir distribuição mínima
    class_counts = df['real_class'].value_counts()
    min_cities = max(5, int(len(df)*0.05))  # Pelo menos 5 cidades ou 5%
    
    for cls, col, q in [('Premium Travel', 'x48', 0.85),
                        ('City Explorer Luxury', 'x40', 0.75)]:
        if class_counts.get(cls, 0) < min_cities:
            candidates = df[df['real_class'] == 'Mid-range Nomad']
            if not candidates.empty:
                threshold = candidates[col].quantile(q)
                df.loc[(df['real_class'] == 'Mid-range Nomad') & 
                       (df[col] >= threshold), 'real_class'] = cls
    
    return df

def main():
    # Aplicar o tema escuro do Matplotlib
    plt.style.use('dark_background')
    
    st.title("Sistema de Classificação Supervisionada de Custos de Viagem")
    
    try:
        # Carregar e preparar dados
        df = pd.read_csv("data_cleaned.csv")
        numeric_cols = [col for col in df.columns if col.startswith('x') and col not in ['x55']]
        
        # Processamento de dados
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        df = criar_labels_reais(df)
        
        # Verificação de dados
        if df.empty:
            st.error("Nenhum dado válido encontrado!")
            return

        # Divisão dos dados
        X = df[numeric_cols]
        y = df['real_class']
        
        if X.shape[0] < 100:
            st.warning("Dados insuficientes para treinamento confiável!")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Pré-processamento
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modelo com balanceamento aprimorado
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Avaliação
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Interface
        st.header("Análise do Modelo")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Acurácia Geral", f"{accuracy:.2%}")
        with col2:
            st.metric("Total de Cidades", len(df))
        with col3:
            st.metric("Classes Únicas", len(y.unique()))
        
        # Distribuição de classes
        st.subheader("Distribuição das Classes")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        class_dist = y.value_counts()
        sns.barplot(x=class_dist.index, y=class_dist.values, ax=ax1, order=class_dist.index, palette='viridis')
        for p in ax1.patches:
            ax1.annotate(f'{p.get_height()}\n({p.get_height()/len(df):.1%})', 
                        (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', color='white')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
        # Matriz de confusão
        st.subheader("Matriz de Confusão Detalhada")
        fig2, ax2 = plt.subplots(figsize=(10,8))
        cm = confusion_matrix(y_test, y_pred, labels=class_dist.index)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                   xticklabels=class_dist.index, yticklabels=class_dist.index, ax=ax2)
        ax2.set_xlabel("Previsto")
        ax2.set_ylabel("Real")
        st.pyplot(fig2)
        
        # Feature importance
        st.subheader("Features Mais Importantes")
        feat_importances = pd.Series(model.feature_importances_, index=numeric_cols)
        top_features = feat_importances.nlargest(10)
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax3, palette='viridis')
        plt.title('Top 10 Features que Impactam a Classificação')
        st.pyplot(fig3)
        

        st.write("O classificador escolhido é o RandomForestClassifier para o aprendizado de máquina baseado em múltiplas árvores de decisão, que melhora a precisão e reduz o overfitting combinando previsões de várias árvores")

        # Simulador Interativo
        st.header("Simulador de Custos de Viagem")
        with st.expander("Configurar Parâmetros", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                x48 = st.slider("Aluguel Centro (x48)", 
                               min_value=0.0, 
                               max_value=df['x48'].max()*2,
                               value=df['x48'].median())
                x54 = st.slider("Salário Médio (x54)", 
                               min_value=0.0, 
                               max_value=df['x54'].max()*2,
                               value=df['x54'].median())
                x40 = st.slider("Atividade Cultural (x40)", 
                               min_value=0.0, 
                               max_value=df['x40'].max()*2,
                               value=df['x40'].median())
            
            with col2:
                x1 = st.slider("Refeição Econômica (x1)", 
                              min_value=0.0, 
                              max_value=df['x1'].max()*2,
                              value=df['x1'].median())
                x41 = st.slider("Entretenimento (x41)", 
                               min_value=0.0, 
                               max_value=df['x41'].max()*2,
                               value=df['x41'].median())
                x38 = st.slider("Internet Veloz (x38)", 
                               min_value=0.0, 
                               max_value=df['x38'].max()*2,
                               value=df['x38'].median())
        
        if st.button("▶️ Prever Classificação"):
            try:
                # Criar dados de entrada
                input_data = pd.DataFrame(
                    np.zeros((1, len(numeric_cols))),
                    columns=numeric_cols
                )
                
                # Mapear sliders para as colunas corretas
                input_mapping = {
                    'x48': x48,
                    'x54': x54,
                    'x40': x40,
                    'x1': x1,
                    'x41': x41,
                    'x38': x38
                }
                
                for col, value in input_mapping.items():
                    input_data[col] = value
                
                # Pré-processar e prever
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled).max()
                
                # Exibir resultados
                st.success(f"**Classificação:** {prediction} | **Confiança:** {proba:.2%}")
                st.markdown("""
                **Legenda de Classes:**
                - 🏨 **Premium Travel:** Alto padrão em todas as categorias
                - 🎭 **City Explorer Luxury:** Excelente infraestrutura cultural/lazer
                - 💼 **Mid-range Nomad:** Equilíbrio entre custo e qualidade
                - 🎒 **Backpacker Budget:** Custo mínimo essencial
                """)
                
            except Exception as e:
                st.error(f"Erro na predição: {str(e)}")

    except FileNotFoundError:
        st.error("Arquivo 'data_cleaned.csv' não encontrado!")
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")



if __name__ == "__main__":
    main()