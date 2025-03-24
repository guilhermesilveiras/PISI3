import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configuração inicial
st.set_page_config(page_title="Classificador de Custos de Viagem")


class_mapping = {
    'Premium Travel': '🏨 Premium Travel',
    'City Explorer Luxury': '🎭 City Explorer Luxury',
    'Mid-range Nomad': '💼 Mid-range Nomad',
    'Backpacker Budget': '🎒 Backpacker Budget' }


tab1, tab2, tab3 = st.tabs(["Classificação", "Normalização e Encoding dos Dados", "Análise SHAP"])

with tab1:

    def criar_labels_reais(df):
        """Cria classificação real com distribuição balanceada"""
        p_aluguel = df['x48'].quantile([0.85, 0.95]).values
        p_salario = df['x54'].quantile([0.80, 0.90]).values
        p_lazer = (df['x40'] + df['x41']).quantile([0.75, 0.90]).values
        
        conditions = [
            (df['x48'] >= p_aluguel[0]) & (df['x54'] >= p_salario[0]),
            ((df['x40'] + df['x41']) >= p_lazer[0]) & (df['x38'] < df['x38'].median()),
            (df['x1'] <= df['x1'].quantile(0.25)) & (df['x28'] <= df['x28'].quantile(0.25)),
        ]
        
        choices = ['Premium Travel', 'City Explorer Luxury', 'Backpacker Budget']
        
        df['real_class'] = np.select(conditions, choices, default='Mid-range Nomad').astype(str)
        
        class_counts = df['real_class'].value_counts()
        min_cities = max(5, int(len(df)*0.05))
        
        for cls, col, q in [('Premium Travel', 'x48', 0.85), ('City Explorer Luxury', 'x40', 0.75)]:
            if class_counts.get(cls, 0) < min_cities:
                candidates = df[df['real_class'] == 'Mid-range Nomad']
                if not candidates.empty:
                    threshold = candidates[col].quantile(q)
                    df.loc[(df['real_class'] == 'Mid-range Nomad') & (df[col] >= threshold), 'real_class'] = cls
        return df

    def main():
        plt.style.use('default')  # Use estilo padrão para fundo claro
        st.title("🌍 Classificação de padrões de gastos em viagens")
        
        st.markdown("""
        **Bem-vindo ao sistema de classificação de custos de viagem!**
        Esta ferramenta utiliza machine learning para classificar cidades em diferentes categorias de custo de vida para viajantes,
        com base em indicadores econômicos e de qualidade de vida.
        """)

        try:
            # Carregar e preparar dados
            df = pd.read_csv("data_cleaned.csv")
            numeric_cols = [col for col in df.columns if col.startswith('x') and col not in ['x55']]
            
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=numeric_cols, how='all').reset_index(drop=True)
            df = df.dropna(thresh=len(numeric_cols)//2).reset_index(drop=True)
            df = criar_labels_reais(df)

            if df.empty:
                st.error("Nenhum dado válido encontrado!")
                return

            # Divisão dos dados
            X = df[numeric_cols]
            y = df['real_class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            # Pipeline de pré-processamento
            preprocessor = make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler()
            )

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Modelos de classificação
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
                'Logistic Regression': LogisticRegression(multi_class='ovr', class_weight='balanced', max_iter=1000, random_state=42)
            }

            trained_models = {}
            model_results = []

            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Coletar métricas por classe
                    for class_name in report:
                        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                            continue
                        
                        # Traduzir nome da classe
                        display_class = class_mapping.get(class_name, class_name)
                        
                        model_results.append({
                            'Modelo': name,
                            'Classe': display_class,
                            'Precisão': f"{report[class_name]['precision']:.2%}",
                            'recall': f"{report[class_name]['recall']:.2%}",
                            'F1-Score': f"{report[class_name]['f1-score']:.2%}",
                            'Suporte': int(report[class_name]['support'])
                        })
                    
                    # Adicionar médias
                    model_results.append({
                        'Modelo': name,
                        'Classe': 'Média Macro',
                        'Precisão': f"{report['macro avg']['precision']:.2%}",
                        'recall': f"{report['macro avg']['recall']:.2%}",
                        'F1-Score': f"{report['macro avg']['f1-score']:.2%}",
                        'Suporte': ''
                    })
                    
                    model_results.append({
                        'Modelo': name,
                        'Classe': 'Média Ponderada',
                        'Precisão': f"{report['weighted avg']['precision']:.2%}",
                        'recall': f"{report['weighted avg']['recall']:.2%}",
                        'F1-Score': f"{report['weighted avg']['f1-score']:.2%}",
                        'Suporte': int(report['weighted avg']['support'])
                    })

                    trained_models[name] = model
                    
                except Exception as e:
                    st.error(f"Erro no modelo {name}: {str(e)}")

            # Seção de Análise de Modelos
            st.header("📈 Análise de Desempenho dos Modelos")
            
            with st.expander("🔍 Métricas Comparativas", expanded=True):
            
                st.markdown("### Tabela de Métricas")
                results_df = pd.DataFrame(model_results)[['Modelo', 'Classe', 'Precisão', 'recall', 'F1-Score', 'Suporte']]
                st.dataframe(results_df, height=500)

            # Seção de Visualizações
            tab1, tab3 = st.tabs(["📊 Matriz de Confusão", "🎮 Simulador"])

            with tab1:
                st.header("Matriz de Confusão")
                selected_model_conf = st.selectbox("Selecione o modelo:", list(trained_models.keys()), key='conf_matrix')
                
                model = trained_models[selected_model_conf]
                y_pred = model.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=model.classes_, 
                            yticklabels=model.classes_, 
                            ax=ax)
                ax.set_xlabel("Previsto")
                ax.set_ylabel("Real")
                st.pyplot(fig)


            with tab3:
                st.header("Simulador de Classificação")
                selected_model = st.selectbox('Selecione o modelo:', list(trained_models.keys()), key='simulator')
                
                with st.expander("⚙️ Configurar Parâmetros", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        x48 = st.slider("Aluguel Centro (x48)", df['x48'].min(), df['x48'].max()*2, df['x48'].median())
                        x54 = st.slider("Salário Médio (x54)", df['x54'].min(), df['x54'].max()*2, df['x54'].median())
                        x40 = st.slider("Atividade Cultural (x40)", df['x40'].min(), df['x40'].max()*2, df['x40'].median())
                    
                    with col2:
                        x1 = st.slider("Refeição Econômica (x1)", df['x1'].min(), df['x1'].max()*2, df['x1'].median())
                        x41 = st.slider("Entretenimento (x41)", df['x41'].min(), df['x41'].max()*2, df['x41'].median())
                        x38 = st.slider("Internet Veloz (x38)", df['x38'].min(), df['x38'].max()*2, df['x38'].median())
                
                if st.button("▶️ Executar Previsão"):
                    try:
                        input_data = pd.DataFrame(np.zeros((1, len(numeric_cols))), columns=numeric_cols)
                        input_mapping = {'x48': x48, 'x54': x54, 'x40': x40, 'x1': x1, 'x41': x41, 'x38': x38}
                        
                        for col, value in input_mapping.items():
                            input_data[col] = value
                        
                        input_scaled = preprocessor.transform(input_data)
                        model = trained_models[selected_model]
                        prediction = model.predict(input_scaled)[0]
                        proba = model.predict_proba(input_scaled).max()
                        
                        st.success(f"**Resultado ({selected_model}):** {prediction} | **Confiança:** {proba:.2%}")
                        st.markdown("""
                        **Legenda de Classes:**
                        - 🏨 **Premium Travel:** Alto padrão
                        - 🎭 **City Explorer Luxury:** Cultura/lazer
                        - 💼 **Mid-range Nomad:** Equilíbrio
                        - 🎒 **Backpacker Budget:** Econômico
                        """)
                        
                    except Exception as e:
                        st.error(f"Erro na predição: {str(e)}")

            # Análise Exploratória
            st.header("🔎 Análise dos Dados")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribuição das Classes")
                fig1, ax1 = plt.subplots(figsize=(10,6))
                class_dist = y.value_counts()
                sns.barplot(x=class_dist.index, y=class_dist.values, ax=ax1, palette='viridis')
                plt.xticks(rotation=45)
                st.pyplot(fig1)
            
            with col2:
                st.markdown("### Features Importantes (RF)")
                rf_model = trained_models['Random Forest']
                feat_importances = pd.Series(rf_model.feature_importances_, index=numeric_cols)
                top_features = feat_importances.nlargest(10)
                fig3, ax3 = plt.subplots(figsize=(10,6))
                sns.barplot(x=top_features.values, y=top_features.index, ax=ax3, palette='viridis')
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"Erro crítico: {str(e)}")

    if __name__ == "__main__":
        main()

with tab2:

    # carregando o arquivo
    data = pd.read_csv('data.csv')

    # selecionando as colunas para normalização e renomeando-as
    columns_to_normalize = {
        'x2': 'Refeição 2 pessoas',
        'x3': 'McMeal',
        'x28': 'Tarifa Táxi',
        'x30': 'Gasolina (1L)',
        'x33': 'Ingresso Cinema',
        'x41': 'Apartamento Centro da Cidade',
        'x48': 'Apartamento 3 Quartos Centro',
        'x50': 'Apartamento 3 Quartos Fora do Centro'
    }

    # Criando DataFrame com as colunas selecionadas
    X = data[list(columns_to_normalize.keys())].copy()
    X.columns = list(columns_to_normalize.values())
    X.insert(0, 'Cidade', data['city'].values)

    # Normalizando os dados para todas as cidades
    scaler = MinMaxScaler()
    X_normalized_values = scaler.fit_transform(X.drop(columns=['Cidade']))
    X_normalized = pd.DataFrame(X_normalized_values, columns=X.columns[1:])
    X_normalized.insert(0, 'Cidade', data['city'].values)

    # Criando um mapa de calor interativo
    st.title("Normalização dos Dados")
    st.write("""
    ### Importância da Normalização
    A normalização garante que todas as features estejam na mesma escala, essencial para algoritmos de aprendizado de máquina que dependem de medidas de distância. A classificação realizada nesse trabalho dependeu diretamente dos dados normalizados.

    ### Passo a Passo da Normalização
    1. **Selecionar Colunas**: Selecionamos as colunas que queremos normalizar. Para a montagem desse dashboard, utilizamos apenas algumas colunas, priorizando aquelas presentes no aplicativo apresentado no artigo.
    2. **Criar DataFrame**: Criamos um DataFrame com as colunas selecionadas e adicionamos a coluna 'Cidade'.
    3. **Normalizar Dados**: Utilizamos o `MinMaxScaler` da biblioteca `sklearn.preprocessing` para normalizar os dados. Este algoritmo transforma os dados para que o menor valor em cada feature seja 0 e o maior valor seja 1.
    4. **Criar DataFrame Normalizado**: Convertendo os dados normalizados de volta para um DataFrame para facilitar a visualização.
    """)

    # Mostrando os dados normalizados no Streamlit
    st.write("### Dados Normalizados:")
    st.dataframe(X_normalized.style.background_gradient(cmap='YlOrRd', axis=0))

    # Selecionando colunas e cidades para exibição no heatmap
    selected_columns = st.multiselect(
        'Selecione as colunas para exibir no heatmap',
        options=list(columns_to_normalize.values()),
        default=list(columns_to_normalize.values())
    )

    # Selecionando cidades para exibição no heatmap
    default_cities = ['Roma', 'Recife', 'Paris', 'New York']
    selected_cities = st.multiselect(
        'Selecione as cidades para exibir no heatmap',
        options=data['city'].unique(),
        default=[city for city in default_cities if city in data['city'].unique()]
    )

    # Filtrando os dados com base na seleção do usuário para o heatmap
    filtered_normalized_data = X_normalized[X_normalized['Cidade'].isin(selected_cities)]

    # Exibindo um heatmap para melhor visualização
    if not filtered_normalized_data.empty:
        st.write("### Heatmap dos Dados Normalizados")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(filtered_normalized_data.set_index('Cidade')[selected_columns], cmap='YlOrRd', annot=True, fmt=".2f", linewidths=.5, ax=ax)
        plt.title('Heatmap dos Dados Normalizados', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.write("Nenhum dado disponível para exibição no heatmap.")

with tab3:
    st.title("Análise SHAP do Modelo Random Forest")
    st.write("""
    ### O que é SHAP?
    SHAP (SHapley Additive exPlanations) é um método baseado na teoria dos jogos que explica as previsões dos modelos de Machine Learning.
    Ele atribui a cada feature uma contribuição para a decisão final do modelo, permitindo entender melhor como os dados afetam a predição.

    ### Gráficos SHAP
    Os gráficos SHAP abaixo mostram a importância de cada feature para o modelo Random Forest, separados pelas classes apresentadas em nossa classificação.
    """)

    # Carregar e preparar dados
    df = pd.read_csv("data_cleaned.csv")
    numeric_cols = [col for col in df.columns if col.startswith('x') and col not in ['x55']]
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols, how='any').reset_index(drop=True)  # Remover cidades com valores nulos
    df = criar_labels_reais(df)

    # Divisão dos dados
    X = df[numeric_cols]
    y = df['real_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Pipeline de pré-processamento
    preprocessor = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Treinar o modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, class_weight='balanced', max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Criar o objeto SHAP explainer
    explainer = shap.Explainer(rf_model, X_train_scaled)
    shap_values = explainer(X_test_scaled, check_additivity=False)

    # Renomear as features
    feature_names = [
        "Meal, Inexpensive Restaurant", "Meal for 2 People", "McMeal at McDonalds", "Domestic Beer",
        "Imported Beer", "Cappuccino", "Coke/Pepsi", "Water", "Milk", "Loaf of Fresh White Bread",
        "Rice", "Eggs", "Local Cheese", "Chicken Fillets", "Beef Round", "Apples", "Banana", "Oranges",
        "Tomato", "Potato", "Onion", "Lettuce", "Water (1.5 liter bottle)", "Bottle of Wine",
        "Domestic Beer (market)", "Imported Beer (market)", "Cigarettes 20 Pack", "One-way Ticket",
        "Monthly Pass", "Taxi Start", "Taxi 1km", "Taxi 1hour Waiting", "Gasoline", "Volkswagen Golf",
        "Toyota Corolla", "Basic Utilities", "Prepaid Mobile Tariff", "Internet", "Fitness Club",
        "Tennis Court Rent", "Cinema", "Preschool", "International Primary School", "1 Pair of Jeans",
        "1 Summer Dress", "1 Pair of Nike Running Shoes", "1 Pair of Men Leather Business Shoes",
        "Apartment (1 bedroom, City Centre)", "Apartment (1 bedroom, Outside Centre)", "Apartment (3 bedrooms, City Centre)",
        "Apartment (3 bedrooms, Outside Centre)", "Price per Square Meter (City Centre)", "Price per Square Meter (Outside Centre)",
        "Average Monthly Net Salary", "Mortgage Interest Rate"
    ]

    # Excluir a feature 0
    shap_values = shap_values[:, 1:]
    X_test_scaled = X_test_scaled[:, 1:]

    # Plotar gráficos SHAP para cada classe
    classes = rf_model.classes_
    for i, class_name in enumerate(classes):
        st.write(f"### Gráfico SHAP para a classe: {class_name}")
        plt.style.use('default')  # Use estilo padrão para fundo claro
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, i], X_test_scaled, feature_names=feature_names, plot_type="dot", show=False)
        plt.title(f'Gráfico SHAP do Modelo Random Forest - Classe: {class_name}', fontsize=16)
        plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        st.pyplot(fig)

    # Explicação das classes
    st.write("""
    As classes presentes nas métricas do SHAP representam diferentes categorias de custo de vida para viajantes:
    
    - **Premium Travel**: Alto padrão de vida, com altos custos de aluguel e salário médio.
    - **City Explorer Luxury**: Foco em atividades culturais e de lazer, com custos elevados nessas áreas.
    - **Mid-range Nomad**: Equilíbrio entre custo e qualidade de vida, sem extremos em nenhuma categoria.
    - **Backpacker Budget**: Foco em economia, com baixos custos em alimentação e transporte.
    """)