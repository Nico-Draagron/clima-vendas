# ============================================================================
# 🤖 pages/modelo_preditivo.py - MODELO PREDITIVO INTEGRADO
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar o modelo preditivo existente
try:
    from modelo_preditivo import ModeloVendasBootstrap
    MODEL_AVAILABLE = True
except ImportError:
    st.warning("⚠️ modelo_preditivo.py não encontrado. Usando funcionalidade limitada.")
    MODEL_AVAILABLE = False

class ModeloPreditivoPage:
    """Página completa do modelo preditivo integrado"""
    
    def __init__(self, store_manager, auth_manager):
        self.store_manager = store_manager
        self.auth_manager = auth_manager
        self.modelo = None
    
    def render(self):
        """Renderiza página principal do modelo preditivo"""
        
        st.markdown("# 🤖 Modelo Preditivo de Vendas")
        st.markdown("**Sistema inteligente de previsão baseado em dados climáticos e histórico de vendas**")
        
        # Verificar permissões
        if not self.auth_manager.has_permission('use_models'):
            st.error("❌ Você não tem permissão para usar modelos preditivos.")
            return
        
        # Carregar dados
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.error("❌ Nenhuma loja configurada. Configure uma loja no painel administrativo.")
            return
        
        # Seleção de loja
        store_options = {f"{info['display_name']} ({store_id})": store_id 
                        for store_id, info in stores.items()}
        
        selected_display = st.selectbox(
            "🏪 Escolha uma loja para análise preditiva:",
            options=list(store_options.keys())
        )
        
        selected_store_id = store_options[selected_display]
        
        # Carregar dados da loja
        df = self.store_manager.load_store_data(selected_store_id)
        
        if df is None or df.empty:
            st.error("❌ Não foi possível carregar dados da loja")
            return
        
        store_info = stores[selected_store_id]
        value_col = store_info['value_column']
        
        if value_col not in df.columns:
            st.error(f"❌ Coluna de vendas '{value_col}' não encontrada nos dados")
            return
        
        # Preparar dados
        df = self._prepare_data(df, value_col)
        
        if df is None or len(df) < 30:
            st.error("❌ Dados insuficientes para treinamento (mínimo 30 registros)")
            return
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dados e Preparação",
            "🔧 Treinamento do Modelo",
            "📈 Performance e Métricas", 
            "🔮 Fazer Previsões",
            "💾 Gerenciar Modelos"
        ])
        
        with tab1:
            self._render_data_preparation(df, value_col, store_info['display_name'])
        
        with tab2:
            self._render_model_training(df, value_col)
        
        with tab3:
            self._render_model_performance(df, value_col)
        
        with tab4:
            self._render_predictions(df, value_col)
        
        with tab5:
            self._render_model_management()
    
    def _prepare_data(self, df, value_col):
        """Prepara dados para o modelo preditivo"""
        
        # Converter data
        df['data'] = pd.to_datetime(df['data'])
        
        # Verificar colunas climáticas disponíveis
        climate_cols = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana']
        available_climate = [col for col in climate_cols if col in df.columns]
        
        if len(available_climate) < 2:
            st.error("❌ Dados climáticos insuficientes. Necessárias pelo menos 2 variáveis climáticas.")
            return None
        
        # Remover valores faltantes
        df_clean = df.dropna(subset=[value_col] + available_climate)
        
        if len(df_clean) < len(df) * 0.8:
            st.warning(f"⚠️ Muitos dados faltantes removidos. Dataset reduzido de {len(df)} para {len(df_clean)} registros.")
        
        # Adicionar features temporais
        df_clean = self._add_temporal_features(df_clean)
        
        return df_clean
    
    def _add_temporal_features(self, df):
        """Adiciona features temporais para o modelo"""
        
        df = df.copy()
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_mes'] = df['data'].dt.day
        df['dia_ano'] = df['data'].dt.dayofyear
        df['eh_weekend'] = df['dia_semana'].isin([5, 6]).astype(int)
        df['trimestre'] = df['data'].dt.quarter
        
        # Features cíclicas (importante para capturar sazonalidade)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365)
        df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365)
        
        return df
    
    def _render_data_preparation(self, df, value_col, store_name):
        """Renderiza tab de preparação de dados"""
        
        st.subheader(f"📊 Dados - {store_name}")
        
        # Estatísticas dos dados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Total de Registros", len(df))
        
        with col2:
            st.metric("🌡️ Variáveis Climáticas", len([col for col in df.columns if any(x in col for x in ['temp', 'precip', 'umid', 'rad'])]))
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("❓ Dados Faltantes", f"{missing_pct:.1f}%")
        
        with col4:
            date_range = (df['data'].max() - df['data'].min()).days
            st.metric("📊 Período (dias)", date_range)
        
        # Visualização dos dados
        st.subheader("📈 Visualização dos Dados")
        
        # Série temporal das vendas
        fig_series = px.line(
            df,
            x='data',
            y=value_col,
            title="Série Temporal - Vendas Diárias",
            labels={'data': 'Data', value_col: 'Vendas (R$)'}
        )
        fig_series.update_layout(height=400)
        st.plotly_chart(fig_series, use_container_width=True)
        
        # Distribuições das variáveis
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição das vendas
            fig_dist_vendas = px.histogram(
                df,
                x=value_col,
                nbins=30,
                title="Distribuição das Vendas",
                labels={value_col: 'Vendas (R$)'}
            )
            st.plotly_chart(fig_dist_vendas, use_container_width=True)
        
        with col2:
            # Distribuição da temperatura (se disponível)
            if 'temp_media' in df.columns:
                fig_dist_temp = px.histogram(
                    df,
                    x='temp_media',
                    nbins=20,
                    title="Distribuição da Temperatura",
                    labels={'temp_media': 'Temperatura Média (°C)'}
                )
                st.plotly_chart(fig_dist_temp, use_container_width=True)
        
        # Qualidade dos dados
        st.subheader("🔍 Qualidade dos Dados")
        
        # Verificações de qualidade
        quality_checks = []
        
        # Check 1: Outliers nas vendas
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[value_col] < Q1 - 1.5*IQR) | (df[value_col] > Q3 + 1.5*IQR)]
        
        quality_checks.append({
            'check': 'Outliers nas Vendas',
            'result': f"{len(outliers)} outliers detectados ({len(outliers)/len(df)*100:.1f}%)",
            'status': '✅' if len(outliers) < len(df)*0.05 else '⚠️'
        })
        
        # Check 2: Dados faltantes
        missing_count = df.isnull().sum().sum()
        quality_checks.append({
            'check': 'Dados Faltantes',
            'result': f"{missing_count} valores faltantes ({missing_pct:.1f}%)",
            'status': '✅' if missing_pct < 5 else '⚠️' if missing_pct < 15 else '❌'
        })
        
        # Check 3: Variabilidade das vendas
        cv = df[value_col].std() / df[value_col].mean()
        quality_checks.append({
            'check': 'Variabilidade das Vendas',
            'result': f"Coeficiente de variação: {cv:.3f}",
            'status': '✅' if cv < 0.5 else '⚠️'
        })
        
        # Check 4: Completude temporal
        expected_days = (df['data'].max() - df['data'].min()).days + 1
        actual_days = len(df['data'].unique())
        completeness = actual_days / expected_days
        
        quality_checks.append({
            'check': 'Completude Temporal',
            'result': f"{completeness:.1%} dos dias com dados",
            'status': '✅' if completeness > 0.9 else '⚠️' if completeness > 0.7 else '❌'
        })
        
        # Exibir checks
        for check in quality_checks:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(check['status'])
            with col2:
                st.write(f"**{check['check']}**: {check['result']}")
        
        # Features disponíveis para o modelo
        st.subheader("🎯 Features Disponíveis para o Modelo")
        
        feature_categories = {
            'Climáticas': [col for col in df.columns if any(x in col for x in ['temp', 'precip', 'umid', 'rad'])],
            'Temporais': [col for col in df.columns if any(x in col for x in ['mes', 'dia', 'ano', 'trimestre'])],
            'Cíclicas': [col for col in df.columns if any(x in col for x in ['sin', 'cos'])],
            'Target': [value_col]
        }
        
        for category, features in feature_categories.items():
            if features:
                st.write(f"**{category}** ({len(features)}): {', '.join(features)}")
    
    def _render_model_training(self, df, value_col):
        """Renderiza tab de treinamento do modelo"""
        
        st.subheader("🔧 Treinamento do Modelo")
        
        if not MODEL_AVAILABLE:
            st.error("❌ Módulo modelo_preditivo não disponível. Verifique a instalação.")
            return
        
        # Configurações do modelo
        st.subheader("⚙️ Configurações do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("📊 Tamanho do Conjunto de Teste", 0.1, 0.4, 0.2, 0.05)
            n_estimators = st.selectbox("🌳 Número de Estimadores (Random Forest)", [50, 100, 200, 300], index=1)
            bootstrap_samples = st.selectbox("🔄 Amostras Bootstrap", [100, 200, 500, 1000], index=1)
        
        with col2:
            include_weather = st.checkbox("🌤️ Incluir Variáveis Climáticas", value=True)
            include_temporal = st.checkbox("📅 Incluir Features Temporais", value=True)
            include_cyclical = st.checkbox("🔄 Incluir Features Cíclicas", value=True)
        
        # Seleção manual de features
        if st.checkbox("🎛️ Seleção Manual de Features"):
            st.subheader("📋 Selecionador de Features")
            
            available_features = []
            
            if include_weather:
                weather_cols = [col for col in df.columns if any(x in col for x in ['temp', 'precip', 'umid', 'rad'])]
                available_features.extend(weather_cols)
            
            if include_temporal:
                temporal_cols = ['mes', 'dia_semana', 'dia_mes', 'trimestre', 'eh_weekend']
                temporal_cols = [col for col in temporal_cols if col in df.columns]
                available_features.extend(temporal_cols)
            
            if include_cyclical:
                cyclical_cols = [col for col in df.columns if any(x in col for x in ['sin', 'cos'])]
                available_features.extend(cyclical_cols)
            
            selected_features = st.multiselect(
                "Escolha as features para o modelo:",
                available_features,
                default=available_features[:10] if len(available_features) > 10 else available_features
            )
        else:
            # Auto-seleção de features
            selected_features = []
            
            if include_weather:
                selected_features.extend([col for col in df.columns if any(x in col for x in ['temp', 'precip', 'umid', 'rad'])])
            
            if include_temporal:
                temporal_cols = ['mes', 'dia_semana', 'dia_mes', 'trimestre', 'eh_weekend']
                selected_features.extend([col for col in temporal_cols if col in df.columns])
            
            if include_cyclical:
                selected_features.extend([col for col in df.columns if any(x in col for x in ['sin', 'cos'])])
        
        if not selected_features:
            st.warning("⚠️ Nenhuma feature selecionada. Selecione ao menos uma variável.")
            return
        
        st.write(f"**Features selecionadas ({len(selected_features)})**: {', '.join(selected_features)}")
        
        # Botão de treinamento
        if st.button("🚀 Treinar Modelo", type="primary"):
            
            with st.spinner("Treinando modelo... Isso pode levar alguns minutos."):
                
                try:
                    # Preparar dados para treinamento
                    features_df = df[selected_features].fillna(df[selected_features].mean())
                    target_df = df[value_col]
                    
                    # Remover linhas com NaN no target
                    valid_indices = ~target_df.isna()
                    features_df = features_df[valid_indices]
                    target_df = target_df[valid_indices]
                    
                    if len(features_df) < 20:
                        st.error("❌ Dados insuficientes após limpeza para treinamento.")
                        return
                    
                    # Criar e treinar modelo
                    self.modelo = ModeloVendasBootstrap(
                        n_bootstrap_samples=bootstrap_samples,
                        test_size=test_size,
                        random_state=42
                    )
                    
                    # Adicionar configurações do Random Forest
                    rf_params = {'n_estimators': n_estimators, 'random_state': 42, 'n_jobs': -1}
                    
                    # Treinar modelo
                    relatorio = self.modelo.treinar(
                        features_df, 
                        target_df,
                        modelo_params={'RandomForest': rf_params}
                    )
                    
                    # Salvar modelo treinado na sessão
                    st.session_state['modelo_treinado'] = self.modelo
                    st.session_state['relatorio_modelo'] = relatorio
                    st.session_state['features_modelo'] = selected_features
                    
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    # Exibir resultados do treinamento
                    self._display_training_results(relatorio)
                    
                except Exception as e:
                    st.error(f"❌ Erro durante o treinamento: {str(e)}")
                    st.exception(e)
    
    def _display_training_results(self, relatorio):
        """Exibe resultados do treinamento"""
        
        st.subheader("📊 Resultados do Treinamento")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rmse_mean = relatorio['metricas']['RMSE']['media']
            st.metric("📊 RMSE Médio", f"{rmse_mean:,.2f}")
        
        with col2:
            r2_mean = relatorio['metricas']['R²']['media']
            st.metric("📈 R² Médio", f"{r2_mean:.3f}")
        
        with col3:
            n_samples = relatorio.get('n_bootstrap_samples', 0)
            st.metric("🔄 Amostras Bootstrap", n_samples)
        
        with col4:
            melhor_modelo = relatorio.get('melhor_modelo', 'N/A')
            st.metric("🏆 Melhor Modelo", melhor_modelo)
        
        # Gráfico de métricas bootstrap
        if 'historico_metricas' in relatorio:
            historico = relatorio['historico_metricas']
            
            fig_metrics = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Distribuição RMSE', 'Distribuição R²']
            )
            
            fig_metrics.add_trace(
                go.Histogram(x=historico['RMSE'], name='RMSE', nbinsx=20),
                row=1, col=1
            )
            
            fig_metrics.add_trace(
                go.Histogram(x=historico['R²'], name='R²', nbinsx=20),
                row=1, col=2
            )
            
            fig_metrics.update_layout(
                height=400,
                title_text="Distribuição das Métricas (Bootstrap)",
                showlegend=False
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Importância das features
        if 'importancia_features' in relatorio and relatorio['importancia_features']:
            st.subheader("🎯 Importância das Features")
            
            # Pegar importância do melhor modelo ou do ensemble
            if relatorio.get('melhor_modelo') and relatorio['melhor_modelo'] in relatorio['importancia_features']:
                importancia = relatorio['importancia_features'][relatorio['melhor_modelo']]
            else:
                # Usar primeiro modelo disponível
                first_model = list(relatorio['importancia_features'].keys())[0]
                importancia = relatorio['importancia_features'][first_model]
            
            # Criar gráfico de importância
            features_sorted = sorted(importancia.items(), key=lambda x: x[1], reverse=True)
            features_names = [item[0] for item in features_sorted[:15]]  # Top 15
            features_importance = [item[1] for item in features_sorted[:15]]
            
            fig_importance = px.bar(
                x=features_importance,
                y=features_names,
                orientation='h',
                title="Top 15 Features Mais Importantes",
                labels={'x': 'Importância', 'y': 'Features'}
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def _render_model_performance(self, df, value_col):
        """Renderiza tab de performance do modelo"""
        
        st.subheader("📈 Performance e Métricas")
        
        # Verificar se há modelo treinado
        if 'modelo_treinado' not in st.session_state:
            st.info("ℹ️ Nenhum modelo treinado. Treine um modelo na aba 'Treinamento do Modelo'.")
            return
        
        modelo = st.session_state['modelo_treinado']
        relatorio = st.session_state['relatorio_modelo']
        features = st.session_state['features_modelo']
        
        # Métricas detalhadas
        st.subheader("📊 Métricas Detalhadas")
        
        metricas = relatorio['metricas']
        
        # Criar DataFrame das métricas
        metrics_data = []
        for metric_name, metric_values in metricas.items():
            metrics_data.append({
                'Métrica': metric_name,
                'Média': f"{metric_values['media']:.4f}",
                'Desvio Padrão': f"{metric_values['std']:.4f}",
                'Mínimo': f"{metric_values['min']:.4f}",
                'Máximo': f"{metric_values['max']:.4f}",
                'IC 95% Inferior': f"{metric_values['ic_inferior']:.4f}",
                'IC 95% Superior': f"{metric_values['ic_superior']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Fazer predições nos dados de treino para análise
        st.subheader("🎯 Análise de Predições")
        
        try:
            features_df = df[features].fillna(df[features].mean())
            target_df = df[value_col]
            
            # Remover NaN
            valid_indices = ~target_df.isna()
            features_clean = features_df[valid_indices]
            target_clean = target_df[valid_indices]
            
            # Fazer predições
            predicoes = modelo.prever(features_clean, usar_ensemble=True, retornar_intervalo=False)
            
            # Gráfico scatter: Predito vs Real
            fig_scatter = px.scatter(
                x=target_clean,
                y=predicoes['predicao'],
                title="Predições vs Valores Reais",
                labels={'x': 'Valores Reais (R$)', 'y': 'Predições (R$)'}
            )
            
            # Adicionar linha y=x (predição perfeita)
            min_val = min(target_clean.min(), predicoes['predicao'].min())
            max_val = max(target_clean.max(), predicoes['predicao'].max())
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Predição Perfeita',
                    line=dict(dash='dash', color='red')
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Residuals plot
            residuals = target_clean - predicoes['predicao']
            
            fig_residuals = px.scatter(
                x=predicoes['predicao'],
                y=residuals,
                title="Análise de Resíduos",
                labels={'x': 'Predições (R$)', 'y': 'Resíduos (R$)'}
            )
            
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Estatísticas dos resíduos
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Resíduo Médio", f"{residuals.mean():.2f}")
            
            with col2:
                st.metric("📈 Desvio Padrão", f"{residuals.std():.2f}")
            
            with col3:
                mae = np.mean(np.abs(residuals))
                st.metric("📊 MAE", f"{mae:.2f}")
            
            with col4:
                mape = np.mean(np.abs(residuals / target_clean)) * 100
                st.metric("📊 MAPE", f"{mape:.1f}%")
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar análise de performance: {str(e)}")
    
    def _render_predictions(self, df, value_col):
        """Renderiza tab de previsões"""
        
        st.subheader("🔮 Fazer Previsões")
        
        # Verificar se há modelo treinado
        if 'modelo_treinado' not in st.session_state:
            st.info("ℹ️ Nenhum modelo treinado. Treine um modelo na aba 'Treinamento do Modelo'.")
            return
        
        modelo = st.session_state['modelo_treinado']
        features = st.session_state['features_modelo']
        
        # Opções de previsão
        prediction_type = st.radio(
            "Tipo de previsão:",
            ["📅 Próximos N dias", "🎯 Cenário Específico", "📊 Dados Históricos"]
        )
        
        if prediction_type == "📅 Próximos N dias":
            self._render_future_predictions(modelo, features, df)
            
        elif prediction_type == "🎯 Cenário Específico":
            self._render_scenario_predictions(modelo, features)
            
        elif prediction_type == "📊 Dados Históricos":
            self._render_historical_predictions(modelo, features, df, value_col)
    
    def _render_future_predictions(self, modelo, features, df):
        """Renderiza previsões para próximos dias"""
        
        st.subheader("📅 Previsões Futuras")
        
        # Configurações
        col1, col2 = st.columns(2)
        
        with col1:
            days_ahead = st.slider("Número de dias para prever", 1, 30, 7)
        
        with col2:
            confidence_interval = st.checkbox("📊 Incluir intervalo de confiança", value=True)
        
        # Inputs de dados climáticos futuros
        st.subheader("🌤️ Dados Climáticos Futuros")
        st.write("Informe os dados climáticos previstos ou deixe em branco para usar médias históricas:")
        
        # Identificar features climáticas necessárias
        climate_features = [f for f in features if any(x in f for x in ['temp', 'precip', 'umid', 'rad'])]
        
        future_data = {}
        
        if climate_features:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temp_media' in climate_features:
                    temp_media = st.number_input("🌡️ Temperatura média (°C)", 
                                               value=float(df['temp_media'].mean()) if 'temp_media' in df.columns else 25.0,
                                               min_value=-10.0, max_value=50.0)
                    future_data['temp_media'] = temp_media
                
                if 'temp_max' in climate_features:
                    temp_max = st.number_input("🌡️ Temperatura máxima (°C)", 
                                             value=float(df['temp_max'].mean()) if 'temp_max' in df.columns else 30.0,
                                             min_value=-5.0, max_value=55.0)
                    future_data['temp_max'] = temp_max
            
            with col2:
                if 'precipitacao_total' in climate_features:
                    precip = st.number_input("🌧️ Precipitação (mm)", 
                                           value=float(df['precipitacao_total'].mean()) if 'precipitacao_total' in df.columns else 2.0,
                                           min_value=0.0, max_value=200.0)
                    future_data['precipitacao_total'] = precip
                
                if 'umid_mediana' in climate_features:
                    umid = st.number_input("💧 Umidade mediana (%)", 
                                         value=float(df['umid_mediana'].mean()) if 'umid_mediana' in df.columns else 70.0,
                                         min_value=0.0, max_value=100.0)
                    future_data['umid_mediana'] = umid
        
        # Botão para fazer previsões
        if st.button("🚀 Gerar Previsões Futuras"):
            
            with st.spinner("Gerando previsões..."):
                
                try:
                    # Criar dados futuros
                    last_date = df['data'].max()
                    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days_ahead, freq='D')
                    
                    future_df_list = []
                    
                    for date in future_dates:
                        row_data = {'data': date}
                        
                        # Adicionar features temporais
                        row_data['mes'] = date.month
                        row_data['dia_semana'] = date.dayofweek
                        row_data['dia_mes'] = date.day
                        row_data['dia_ano'] = date.dayofyear
                        row_data['eh_weekend'] = 1 if date.dayofweek >= 5 else 0
                        row_data['trimestre'] = (date.month - 1) // 3 + 1
                        
                        # Features cíclicas
                        row_data['mes_sin'] = np.sin(2 * np.pi * date.month / 12)
                        row_data['mes_cos'] = np.cos(2 * np.pi * date.month / 12)
                        row_data['dia_ano_sin'] = np.sin(2 * np.pi * date.dayofyear / 365)
                        row_data['dia_ano_cos'] = np.cos(2 * np.pi * date.dayofyear / 365)
                        
                        # Adicionar dados climáticos
                        for feature in climate_features:
                            if feature in future_data:
                                row_data[feature] = future_data[feature]
                            elif feature in df.columns:
                                # Usar média histórica
                                row_data[feature] = df[feature].mean()
                            else:
                                row_data[feature] = 0
                        
                        future_df_list.append(row_data)
                    
                    future_df = pd.DataFrame(future_df_list)
                    
                    # Garantir que todas as features necessárias estejam presentes
                    for feature in features:
                        if feature not in future_df.columns:
                            if feature in df.columns:
                                future_df[feature] = df[feature].mean()
                            else:
                                future_df[feature] = 0
                    
                    # Fazer previsões
                    results = modelo.prever(
                        future_df[features], 
                        usar_ensemble=True, 
                        retornar_intervalo=confidence_interval
                    )
                    
                    # Preparar dados para visualização
                    viz_data = {
                        'Data': future_dates,
                        'Previsão': results['predicao']
                    }
                    
                    if confidence_interval:
                        viz_data['IC Inferior'] = results['intervalo_inferior']
                        viz_data['IC Superior'] = results['intervalo_superior']
                    
                    viz_df = pd.DataFrame(viz_data)
                    
                    # Visualizar previsões
                    fig_pred = go.Figure()
                    
                    # Adicionar previsões
                    fig_pred.add_trace(go.Scatter(
                        x=viz_df['Data'],
                        y=viz_df['Previsão'],
                        mode='lines+markers',
                        name='Previsão',
                        line=dict(color='blue', width=3)
                    ))
                    
                    if confidence_interval:
                        # Adicionar intervalo de confiança
                        fig_pred.add_trace(go.Scatter(
                            x=viz_df['Data'],
                            y=viz_df['IC Superior'],
                            mode='lines',
                            name='IC Superior',
                            line=dict(color='lightblue', width=1),
                            showlegend=False
                        ))
                        
                        fig_pred.add_trace(go.Scatter(
                            x=viz_df['Data'],
                            y=viz_df['IC Inferior'],
                            mode='lines',
                            name='IC Inferior',
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.3)',
                            line=dict(color='lightblue', width=1),
                            showlegend=True
                        ))
                    
                    fig_pred.update_layout(
                        title=f"Previsões para os Próximos {days_ahead} Dias",
                        xaxis_title="Data",
                        yaxis_title="Vendas Previstas (R$)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Tabela com previsões
                    st.subheader("📊 Tabela de Previsões")
                    
                    display_df = viz_df.copy()
                    display_df['Data'] = display_df['Data'].dt.strftime('%d/%m/%Y')
                    display_df['Previsão'] = display_df['Previsão'].apply(lambda x: f"R$ {x:,.2f}".replace(',', '.'))
                    
                    if confidence_interval:
                        display_df['IC Inferior'] = display_df['IC Inferior'].apply(lambda x: f"R$ {x:,.2f}".replace(',', '.'))
                        display_df['IC Superior'] = display_df['IC Superior'].apply(lambda x: f"R$ {x:,.2f}".replace(',', '.'))
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Estatísticas das previsões
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("💰 Vendas Médias Previstas", f"R$ {results['predicao'].mean():,.2f}".replace(',', '.'))
                    
                    with col2:
                        st.metric("📈 Vendas Máximas Previstas", f"R$ {results['predicao'].max():,.2f}".replace(',', '.'))
                    
                    with col3:
                        st.metric("📉 Vendas Mínimas Previstas", f"R$ {results['predicao'].min():,.2f}".replace(',', '.'))
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar previsões: {str(e)}")
                    st.exception(e)
    
    def _render_scenario_predictions(self, modelo, features):
        """Renderiza previsões para cenários específicos"""
        
        st.subheader("🎯 Previsão por Cenário")
        
        st.write("Configure um cenário específico para ver a previsão:")
        
        scenario_data = {}
        
        # Organizar inputs por categoria
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🌤️ Condições Climáticas**")
            
            climate_features = [f for f in features if any(x in f for x in ['temp', 'precip', 'umid', 'rad'])]
            
            for feature in climate_features:
                if 'temp_media' in feature:
                    scenario_data[feature] = st.number_input(f"🌡️ {feature}", value=25.0, min_value=-10.0, max_value=50.0, key=f"scenario_{feature}")
                elif 'temp_max' in feature:
                    scenario_data[feature] = st.number_input(f"🌡️ {feature}", value=30.0, min_value=-5.0, max_value=55.0, key=f"scenario_{feature}")
                elif 'temp_min' in feature:
                    scenario_data[feature] = st.number_input(f"🌡️ {feature}", value=20.0, min_value=-15.0, max_value=45.0, key=f"scenario_{feature}")
                elif 'precipitacao' in feature:
                    scenario_data[feature] = st.number_input(f"🌧️ {feature}", value=0.0, min_value=0.0, max_value=200.0, key=f"scenario_{feature}")
                elif 'umid' in feature:
                    scenario_data[feature] = st.number_input(f"💧 {feature}", value=70.0, min_value=0.0, max_value=100.0, key=f"scenario_{feature}")
                elif 'rad' in feature:
                    scenario_data[feature] = st.number_input(f"☀️ {feature}", value=25.0, min_value=0.0, max_value=50.0, key=f"scenario_{feature}")
        
        with col2:
            st.write("**📅 Características Temporais**")
            
            temporal_features = [f for f in features if any(x in f for x in ['mes', 'dia', 'trimestre'])]
            
            for feature in temporal_features:
                if feature == 'mes':
                    scenario_data[feature] = st.selectbox(f"📅 {feature}", list(range(1, 13)), key=f"scenario_{feature}")
                elif feature == 'dia_semana':
                    scenario_data[feature] = st.selectbox(f"📅 {feature}", list(range(7)), 
                                                        format_func=lambda x: ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'][x],
                                                        key=f"scenario_{feature}")
                elif feature == 'trimestre':
                    scenario_data[feature] = st.selectbox(f"📅 {feature}", [1, 2, 3, 4], key=f"scenario_{feature}")
                elif feature == 'eh_weekend':
                    scenario_data[feature] = st.selectbox(f"📅 {feature}", [0, 1], 
                                                        format_func=lambda x: 'Não' if x == 0 else 'Sim',
                                                        key=f"scenario_{feature}")
                else:
                    scenario_data[feature] = st.number_input(f"📅 {feature}", value=15, key=f"scenario_{feature}")
        
        with col3:
            st.write("**🔄 Features Cíclicas**")
            
            cyclical_features = [f for f in features if any(x in f for x in ['sin', 'cos'])]
            
            for feature in cyclical_features:
                if 'mes' in feature:
                    mes_val = scenario_data.get('mes', 6)
                    if 'sin' in feature:
                        scenario_data[feature] = np.sin(2 * np.pi * mes_val / 12)
                    else:
                        scenario_data[feature] = np.cos(2 * np.pi * mes_val / 12)
                    st.write(f"🔄 {feature}: {scenario_data[feature]:.3f}")
                elif 'dia_ano' in feature:
                    dia_ano = scenario_data.get('dia_ano', 180)
                    if 'sin' in feature:
                        scenario_data[feature] = np.sin(2 * np.pi * dia_ano / 365)
                    else:
                        scenario_data[feature] = np.cos(2 * np.pi * dia_ano / 365)
                    st.write(f"🔄 {feature}: {scenario_data[feature]:.3f}")
        
        # Preencher features faltantes com zeros
        for feature in features:
            if feature not in scenario_data:
                scenario_data[feature] = 0
        
        if st.button("🎯 Calcular Previsão do Cenário"):
            
            try:
                # Criar DataFrame com o cenário
                scenario_df = pd.DataFrame([scenario_data])
                
                # Fazer previsão
                result = modelo.prever(scenario_df[features], usar_ensemble=True, retornar_intervalo=True)
                
                # Exibir resultado
                st.subheader("📊 Resultado da Previsão")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Previsão", f"R$ {result['predicao'][0]:,.2f}".replace(',', '.'))
                
                with col2:
                    st.metric("📉 IC Inferior (95%)", f"R$ {result['intervalo_inferior'][0]:,.2f}".replace(',', '.'))
                
                with col3:
                    st.metric("📈 IC Superior (95%)", f"R$ {result['intervalo_superior'][0]:,.2f}".replace(',', '.'))
                
                # Análise do cenário
                st.subheader("💡 Análise do Cenário")
                
                insights = []
                
                # Análise da temperatura
                if 'temp_media' in scenario_data:
                    temp = scenario_data['temp_media']
                    if temp > 30:
                        insights.append("🌡️ **Temperatura alta**: Pode impactar positivamente ou negativamente dependendo do tipo de produto")
                    elif temp < 15:
                        insights.append("🌡️ **Temperatura baixa**: Considere produtos sazonais de inverno")
                
                # Análise da precipitação
                if 'precipitacao_total' in scenario_data:
                    chuva = scenario_data['precipitacao_total']
                    if chuva > 10:
                        insights.append("🌧️ **Dia chuvoso**: Pode afetar fluxo de clientes. Considere estratégias de delivery")
                    elif chuva == 0:
                        insights.append("☀️ **Dia sem chuva**: Condições favoráveis para movimentação")
                
                # Análise temporal
                if 'eh_weekend' in scenario_data and scenario_data['eh_weekend'] == 1:
                    insights.append("📅 **Final de semana**: Padrão de consumo pode ser diferente dos dias úteis")
                
                for insight in insights:
                    st.info(insight)
                
            except Exception as e:
                st.error(f"❌ Erro ao calcular previsão: {str(e)}")
    
    def _render_historical_predictions(self, modelo, features, df, value_col):
        """Renderiza previsões em dados históricos para validação"""
        
        st.subheader("📊 Validação em Dados Históricos")
        
        st.write("Teste o modelo em dados históricos para validar sua precisão:")
        
        # Seleção de período
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "📅 Data Inicial",
                value=df['data'].max().date() - timedelta(days=30),
                min_value=df['data'].min().date(),
                max_value=df['data'].max().date()
            )
        
        with col2:
            end_date = st.date_input(
                "📅 Data Final",
                value=df['data'].max().date(),
                min_value=df['data'].min().date(),
                max_value=df['data'].max().date()
            )
        
        if start_date >= end_date:
            st.error("❌ Data inicial deve ser anterior à data final")
            return
        
        if st.button("🔍 Executar Validação Histórica"):
            
            try:
                # Filtrar dados do período
                period_df = df[
                    (df['data'] >= pd.to_datetime(start_date)) & 
                    (df['data'] <= pd.to_datetime(end_date))
                ].copy()
                
                if period_df.empty:
                    st.error("❌ Nenhum dado encontrado no período selecionado")
                    return
                
                # Preparar dados
                features_df = period_df[features].fillna(period_df[features].mean())
                target_df = period_df[value_col]
                
                # Remover NaN
                valid_indices = ~target_df.isna()
                features_clean = features_df[valid_indices]
                target_clean = target_df[valid_indices]
                dates_clean = period_df['data'][valid_indices]
                
                if len(features_clean) == 0:
                    st.error("❌ Nenhum dado válido no período selecionado")
                    return
                
                # Fazer previsões
                results = modelo.prever(features_clean, usar_ensemble=True, retornar_intervalo=True)
                
                # Calcular métricas
                predictions = results['predicao']
                rmse = np.sqrt(np.mean((target_clean - predictions) ** 2))
                mae = np.mean(np.abs(target_clean - predictions))
                mape = np.mean(np.abs((target_clean - predictions) / target_clean)) * 100
                r2 = 1 - np.sum((target_clean - predictions) ** 2) / np.sum((target_clean - target_clean.mean()) ** 2)
                
                # Exibir métricas
                st.subheader("📊 Métricas de Validação")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 RMSE", f"{rmse:,.2f}")
                
                with col2:
                    st.metric("📊 MAE", f"{mae:,.2f}")
                
                with col3:
                    st.metric("📊 MAPE", f"{mape:.1f}%")
                
                with col4:
                    st.metric("📊 R²", f"{r2:.3f}")
                
                # Gráfico comparativo
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Scatter(
                    x=dates_clean,
                    y=target_clean,
                    mode='lines+markers',
                    name='Valores Reais',
                    line=dict(color='blue', width=2)
                ))
                
                fig_comparison.add_trace(go.Scatter(
                    x=dates_clean,
                    y=predictions,
                    mode='lines+markers',
                    name='Previsões',
                    line=dict(color='red', width=2)
                ))
                
                # Adicionar intervalo de confiança
                fig_comparison.add_trace(go.Scatter(
                    x=dates_clean,
                    y=results['intervalo_superior'],
                    mode='lines',
                    name='IC Superior',
                    line=dict(color='lightcoral', width=1),
                    showlegend=False
                ))
                
                fig_comparison.add_trace(go.Scatter(
                    x=dates_clean,
                    y=results['intervalo_inferior'],
                    mode='lines',
                    name='IC Inferior',
                    fill='tonexty',
                    fillcolor='rgba(255, 182, 193, 0.3)',
                    line=dict(color='lightcoral', width=1),
                    showlegend=True
                ))
                
                fig_comparison.update_layout(
                    title="Validação Histórica: Real vs Previsto",
                    xaxis_title="Data",
                    yaxis_title="Vendas (R$)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Análise de acurácia por dia
                accuracy_by_day = []
                for i in range(len(target_clean)):
                    error_pct = abs((target_clean.iloc[i] - predictions[i]) / target_clean.iloc[i]) * 100
                    accuracy_by_day.append(100 - error_pct)
                
                avg_accuracy = np.mean(accuracy_by_day)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎯 Acurácia Média", f"{avg_accuracy:.1f}%")
                
                with col2:
                    days_above_80 = len([acc for acc in accuracy_by_day if acc >= 80])
                    st.metric("✅ Dias com Acurácia >80%", f"{days_above_80}/{len(accuracy_by_day)}")
                
            except Exception as e:
                st.error(f"❌ Erro na validação: {str(e)}")
    
    def _render_model_management(self):
        """Renderiza tab de gerenciamento de modelos"""
        
        st.subheader("💾 Gerenciamento de Modelos")
        
        # Salvar modelo treinado
        if 'modelo_treinado' in st.session_state:
            st.subheader("💾 Salvar Modelo Atual")
            
            model_name = st.text_input(
                "Nome do modelo:",
                value=f"modelo_vendas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if st.button("💾 Salvar Modelo"):
                try:
                    modelo = st.session_state['modelo_treinado']
                    filename = f"{model_name}.pkl"
                    
                    # Salvar usando pickle
                    with open(filename, 'wb') as f:
                        pickle.dump({
                            'modelo': modelo,
                            'features': st.session_state['features_modelo'],
                            'relatorio': st.session_state['relatorio_modelo'],
                            'timestamp': datetime.now(),
                            'version': '1.0'
                        }, f)
                    
                    st.success(f"✅ Modelo salvo como {filename}")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao salvar modelo: {str(e)}")
        
        # Carregar modelo salvo
        st.subheader("📂 Carregar Modelo Salvo")
        
        uploaded_model = st.file_uploader(
            "Escolha um arquivo de modelo (.pkl):",
            type=['pkl']
        )
        
        if uploaded_model and st.button("📂 Carregar Modelo"):
            try:
                model_data = pickle.load(uploaded_model)
                
                st.session_state['modelo_treinado'] = model_data['modelo']
                st.session_state['features_modelo'] = model_data['features']
                st.session_state['relatorio_modelo'] = model_data['relatorio']
                
                st.success(f"✅ Modelo carregado com sucesso!")
                st.write(f"**Criado em:** {model_data.get('timestamp', 'N/A')}")
                st.write(f"**Features:** {len(model_data['features'])} variáveis")
                st.write(f"**Versão:** {model_data.get('version', 'N/A')}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        
        # Informações do modelo atual
        if 'modelo_treinado' in st.session_state:
            st.subheader("ℹ️ Informações do Modelo Atual")
            
            relatorio = st.session_state['relatorio_modelo']
            features = st.session_state['features_modelo']
            
            info_data = {
                'Métrica': ['Número de Features', 'RMSE Médio', 'R² Médio', 'Amostras Bootstrap', 'Melhor Modelo'],
                'Valor': [
                    len(features),
                    f"{relatorio['metricas']['RMSE']['media']:.2f}",
                    f"{relatorio['metricas']['R²']['media']:.3f}",
                    relatorio.get('n_bootstrap_samples', 'N/A'),
                    relatorio.get('melhor_modelo', 'N/A')
                ]
            }
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        # Exportar relatório
        if 'relatorio_modelo' in st.session_state:
            st.subheader("📊 Exportar Relatório")
            
            if st.button("📥 Gerar Relatório JSON"):
                relatorio = st.session_state['relatorio_modelo']
                features = st.session_state['features_modelo']
                
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'features_utilizadas': features,
                    'metricas': relatorio['metricas'],
                    'melhor_modelo': relatorio.get('melhor_modelo'),
                    'importancia_features': relatorio.get('importancia_features', {})
                }
                
                import json
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                
                st.download_button(
                    "💾 Download Relatório",
                    data=json_str,
                    file_name=f"relatorio_modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Função para integrar com streamlit_app.py
def show_modelo_preditivo_page(df, role, store_manager, auth_manager):
    """Função para mostrar a página do modelo preditivo"""
    
    page = ModeloPreditivoPage(store_manager, auth_manager)
    page.render()