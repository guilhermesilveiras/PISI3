import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial
st.set_page_config(page_title="Classificador de Custos de Viagem", layout="wide")

def criar_labels_reais(df):
    """Creates real classification based on cost ranges"""
    # Calculate total cost using relevant features
    cost_features = ['x1', 'x48', 'x54']  # Example cost-related features
    total_cost = df[cost_features].sum(axis=1)
    
    # Create labels based on quartiles
    labels = pd.qcut(total_cost, q=4, labels=[
        "Backpacker Budget",
        "Mid-range Nomad",
        "City Explorer Luxury",
        "Premium Travel"
    ])
    
    df['real_class'] = labels
    return df

def treinar_avaliar_modelo(nome_modelo, modelo, X_train_scaled, y_train, X_test_scaled, y_test, class_dist):
    """Treina e avalia um modelo de classificação com validação cruzada"""
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=cv)
    
    # Treinar modelo final
    modelo.fit(X_train_scaled, y_train)
    
    # Avaliação
    y_pred = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=class_dist.index)
    
    # Feature importance (apenas para Random Forest)
    feat_importances = getattr(modelo, 'feature_importances_', None)
    
    return {
        'nome': nome_modelo,
        'modelo': modelo,
        'predicoes': y_pred,
        'acuracia': accuracy,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'report': class_report,
        'conf_matrix': cm,
        'feat_importances': feat_importances
    }

def exibir_metricas(resultados, X, y, numeric_cols, df):
    """Exibe métricas e gráficos para o modelo selecionado"""
    nome_modelo = resultados['nome']
    accuracy = resultados['acuracia']
    class_report = resultados['report']
    cm = resultados['conf_matrix']
    feat_importances = resultados['feat_importances']
    
    st.header(f"Análise do Modelo: {nome_modelo}")
    
    # Métricas gerais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acurácia Geral", f"{accuracy:.2%}")
    with col2:
        st.metric("Total de Cidades", len(df))
    with col3:
        st.metric("Classes Únicas", len(y.unique()))
    
    # Matriz de confusão
    st.subheader("Matriz de Confusão")
    fig_cm, ax_cm = plt.subplots(figsize=(10,8))
    class_dist = y.value_counts()
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
               xticklabels=class_dist.index, yticklabels=class_dist.index, ax=ax_cm)
    ax_cm.set_xlabel("Previsto")
    ax_cm.set_ylabel("Real")
    st.pyplot(fig_cm)
    
    # Tabela detalhada de métricas
    st.subheader("Métricas Detalhadas por Classe")
    metrics_df = pd.DataFrame()
    
    # Por classe
    for cls in class_dist.index:
        if cls in class_report:
            cls_metrics = class_report[cls]
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Classe': [cls],
                'Precisão': [cls_metrics['precision']],
                'Recall': [cls_metrics['recall']],
                'F1-score': [cls_metrics['f1-score']],
                'Suporte': [cls_metrics['support']]
            })], ignore_index=True)
    
    # Adicionar média simples
    metrics_df = pd.concat([metrics_df, pd.DataFrame({
        'Classe': ['Média Simples'],
        'Precisão': [metrics_df['Precisão'].mean()],
        'Recall': [metrics_df['Recall'].mean()],
        'F1-score': [metrics_df['F1-score'].mean()],
        'Suporte': [metrics_df['Suporte'].sum() / len(metrics_df)]
    })], ignore_index=True)
    
    # Adicionar médias do relatório
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in class_report:
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Classe': [avg_type.replace('avg', 'Média').title()],
                'Precisão': [class_report[avg_type]['precision']],
                'Recall': [class_report[avg_type]['recall']],
                'F1-score': [class_report[avg_type]['f1-score']],
                'Suporte': [class_report[avg_type]['support']]
            })], ignore_index=True)
    
    # Formatação para apresentação
    formatted_metrics = metrics_df.copy()
    for col in ['Precisão', 'Recall', 'F1-score']:
        formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2%}")
    formatted_metrics['Suporte'] = formatted_metrics['Suporte'].astype(int)
    
    # Exibir tabela
    st.table(formatted_metrics)
    
    # Exportar métricas
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"📊 Baixar Métricas ({nome_modelo}) CSV",
        csv,
        f"metricas_{nome_modelo.lower().replace(' ', '_')}.csv",
        "text/csv",
        key=f'download-metrics-{nome_modelo.lower().replace(" ", "_")}'
    )
    
    # Feature importance (apenas para Random Forest)
    if feat_importances is not None:
        st.subheader("Features Mais Importantes")
        feat_imp_series = pd.Series(feat_importances, index=numeric_cols)
        top_features = feat_imp_series.nlargest(10)
        fig_imp, ax_imp = plt.subplots(figsize=(10,6))
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax_imp, palette='viridis')
        plt.title('Top 10 Features que Impactam a Classificação')
        st.pyplot(fig_imp)

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

        # Definir os modelos
        modelos = {
            "Random Forest": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            "SVM": SVC(
                kernel='rbf',
                C=10.0,
                gamma='auto',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            "Naive Bayes": GaussianNB(
                var_smoothing=1e-9
            )
        }
        
        # Treinar e avaliar modelos
        class_dist = y.value_counts()
        resultados_modelos = {}
        
        for nome, modelo in modelos.items():
            resultados_modelos[nome] = treinar_avaliar_modelo(
                nome, modelo, X_train_scaled, y_train, X_test_scaled, y_test, class_dist
            )
        
        # Comparação de acurácia entre modelos
        st.header("Comparação de Modelos")
        
        # Gráfico de comparação
        accuracies = {nome: res['acuracia'] for nome, res in resultados_modelos.items()}
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
        for i, v in enumerate(accuracies.values()):
            ax_comp.text(i, v + 0.01, f"{v:.2%}", ha='center')
        ax_comp.set_ylim(0, 1.0)
        ax_comp.set_title("Comparação de Acurácia entre Modelos")
        ax_comp.set_ylabel("Acurácia")
        st.pyplot(fig_comp)
        
        # Comparação de F1-score por classe
        f1_por_modelo = {}
        classes = list(class_dist.index)
        
        for nome, res in resultados_modelos.items():
            f1_scores = []
            for cls in classes:
                if cls in res['report']:
                    f1_scores.append(res['report'][cls]['f1-score'])
            f1_por_modelo[nome] = f1_scores
        
        f1_df = pd.DataFrame(f1_por_modelo, index=classes)
        
        st.subheader("Comparação de F1-Score por Classe")
        fig_f1, ax_f1 = plt.subplots(figsize=(12, 8))
        f1_df.plot(kind='bar', ax=ax_f1)
        ax_f1.set_ylim(0, 1.0)
        ax_f1.set_xlabel("Classe")
        ax_f1.set_ylabel("F1-Score")
        ax_f1.legend(title="Modelo")
        st.pyplot(fig_f1)
        
        # Seleção de modelo para análise detalhada
        modelo_selecionado = st.selectbox(
            "Selecione um modelo para análise detalhada:",
            list(resultados_modelos.keys())
        )
        
        # Exibir análise detalhada do modelo selecionado
        exibir_metricas(
            resultados_modelos[modelo_selecionado], 
            X, y, numeric_cols, df
        )
        
        # Descricão dos Modelos
        st.header("Descrição dos Modelos de Classificação")
        
        st.markdown("""
        ### Random Forest
        O Random Forest cria múltiplas árvores de decisão e combina os resultados para classificação. 
        Vantagens: Funciona bem com muitas features, captura relações não-lineares e interage bem com outliers.
        
        ### Support Vector Machine (SVM)
        O SVM busca encontrar um hiperplano que separe melhor as classes no espaço de features.
        Vantagens: Eficaz em espaços de alta dimensão, utiliza kernel para transformações de features.
        
        ### Naive Bayes
        Classificador probabilístico baseado no teorema de Bayes com a suposição de independência entre features.
        Vantagens: Simples, rápido e requer pouco treinamento (funciona bem mesmo com datasets menores).
        """)
        
        # Simulador Interativo (usando o melhor modelo baseado na acurácia)
        melhor_modelo_nome = max(accuracies, key=accuracies.get)
        melhor_modelo = resultados_modelos[melhor_modelo_nome]['modelo']
        
        st.header(f"Simulador de Custos de Viagem (usando {melhor_modelo_nome})")
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
        
        if st.button(f"▶️ Prever Classificação ({melhor_modelo_nome})"):
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
                prediction = melhor_modelo.predict(input_scaled)[0]
                
                # Obter probabilidades
                if hasattr(melhor_modelo, 'predict_proba'):
                    probas = melhor_modelo.predict_proba(input_scaled)[0]
                    proba = probas.max()
                    prediction = melhor_modelo.classes_[probas.argmax()]
                    
                    # Aplicar threshold
                    confidence_threshold = 0.6  # 60% de confiança mínima
                    if proba < confidence_threshold:
                        st.warning(f"Baixa confiança na previsão ({proba:.2%}). Considere esta classificação como incerta.")
                        # Opcionalmente, mostrar top 2 classes
                        top2_indices = probas.argsort()[-2:][::-1]
                        top2_classes = melhor_modelo.classes_[top2_indices]
                        top2_probas = probas[top2_indices]
                        st.info(f"Possíveis classificações: {top2_classes[0]} ({top2_probas[0]:.2%}) ou {top2_classes[1]} ({top2_probas[1]:.2%})")
                else:
                    proba = 0.0
                
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

        # Resultados da Validação Cruzada
        st.header("Resultados da Validação Cruzada")
        cv_results = {}
        for nome, res in resultados_modelos.items():
            cv_results[nome] = {
                'Média': res['cv_mean'],
                'Desvio Padrão': res['cv_std']
            }
        
        cv_df = pd.DataFrame(cv_results).T
        cv_df.columns = ['Média CV', 'Desvio Padrão CV']
        cv_df = cv_df.round(4)
        st.table(cv_df)

    except FileNotFoundError:
        st.error("Arquivo 'data_cleaned.csv' não encontrado!")
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")

if __name__ == "__main__":
    main()