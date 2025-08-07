# ============================================================================
# 📈 pages/serie_temporal.py - ANÁLISE DE SÉRIE TEMPORAL
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SerieTemporalPage:
    """Página completa de análise de série temporal"""
    
    def __init__(self, store_manager):
        self.store_manager = store_manager
    
    def render(self):
        """Renderiza página principal de série temporal"""
        
        st.markdown("# 📈 Análise de Série Temporal")
        st.markdown("**Análise detalhada dos padrões temporais nas vendas e identificação de tendências e sazonalidade**")
        
        # Carregar dados
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.error("❌ Nenhuma loja configurada. Configure uma loja no painel administrativo.")
            return
        
        # Seleção de loja
        store_options = {f"{info['display_name']} ({store_id})": store_id 
                        for store_id, info in stores.items()}
        
        selected_display = st.selectbox(
            "🏪 Escolha uma loja para análise temporal:",
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
        df = self._prepare_time_series_data(df, value_col)
        
        if df is None or len(df) < 30:
            st.error("❌ Dados insuficientes para análise temporal (mínimo 30 observações)")
            return
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visão Geral",
            "🔍 Decomposição",
            "📈 Tendências",
            "🔄 Sazonalidade",
            "📉 Autocorrelação"
        ])
        
        with tab1:
            self._render_overview(df, value_col, store_info['display_name'])
        
        with tab2:
            self._render_decomposition(df, value_col)
        
        with tab3:
            self._render_trend_analysis(df, value_col)
        
        with tab4:
            self._render_seasonality_analysis(df, value_col)
        
        with tab5:
            self._render_autocorrelation_analysis(df, value_col)
    
    def _prepare_time_series_data(self, df, value_col):
        """Prepara dados para análise de série temporal"""
        
        # Converter data
        df['data'] = pd.to_datetime(df['data'])
        
        # Ordenar por data
        df = df.sort_values('data')
        
        # Verificar se há dados duplicados por data
        duplicates = df.duplicated(subset=['data']).sum()
        if duplicates > 0:
            st.warning(f"⚠️ {duplicates} datas duplicadas encontradas. Agregando por soma.")
            df = df.groupby('data').agg({
                value_col: 'sum',
                **{col: 'mean' for col in df.columns if col not in ['data', value_col] and df[col].dtype in ['float64', 'int64']}
            }).reset_index()
        
        # Verificar continuidade temporal
        date_range = pd.date_range(df['data'].min(), df['data'].max(), freq='D')
        missing_dates = set(date_range) - set(df['data'])
        
        if missing_dates:
            st.info(f"ℹ️ {len(missing_dates)} datas faltantes no período. Preenchendo com interpolação.")
            
            # Criar série completa
            complete_df = pd.DataFrame({'data': date_range})
            df = complete_df.merge(df, on='data', how='left')
            
            # Interpolação linear para vendas
            df[value_col] = df[value_col].interpolate(method='linear')
        
        # Adicionar features temporais
        df = self._add_time_features(df)
        
        return df
    
    def _add_time_features(self, df):
        """Adiciona features temporais ao dataset"""
        
        df = df.copy()
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia'] = df['data'].dt.day
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_ano'] = df['data'].dt.dayofyear
        df['semana'] = df['data'].dt.isocalendar().week
        df['trimestre'] = df['data'].dt.quarter
        df['eh_weekend'] = df['dia_semana'].isin([5, 6])
        
        # Nomes dos dias e meses
        df['nome_dia'] = df['dia_semana'].map({
            0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta',
            4: 'Sexta', 5: 'Sábado', 6: 'Domingo'
        })
        
        df['nome_mes'] = df['mes'].map({
            1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
            5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
            9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
        })
        
        return df
    
    def _render_overview(self, df, value_col, store_name):
        """Renderiza visão geral da série temporal"""
        
        st.subheader(f"📊 Série Temporal - {store_name}")
        
        # Estatísticas básicas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Total de Observações", len(df))
        
        with col2:
            period_days = (df['data'].max() - df['data'].min()).days
            st.metric("📊 Período (dias)", period_days)
        
        with col3:
            mean_sales = df[value_col].mean()
            st.metric("💰 Vendas Médias", f"R$ {mean_sales:,.2f}".replace(',', '.'))
        
        with col4:
            cv = df[value_col].std() / df[value_col].mean()
            st.metric("📈 Coef. Variação", f"{cv:.3f}")
        
        # Série temporal principal
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=df[value_col],
            mode='lines',
            name='Vendas',
            line=dict(color='blue', width=1)
        ))
        
        # Adicionar média móvel
        window_size = min(30, len(df) // 10)
        if window_size >= 7:
            df['media_movel'] = df[value_col].rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(
                x=df['data'],
                y=df['media_movel'],
                mode='lines',
                name=f'Média Móvel ({window_size}d)',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title="Série Temporal de Vendas",
            xaxis_title="Data",
            yaxis_title="Vendas (R$)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas descritivas
        st.subheader("📊 Estatísticas Descritivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stats_data = {
                'Métrica': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 
                           'Q1 (25%)', 'Q3 (75%)', 'Assimetria', 'Curtose'],
                'Valor': [
                    f"R$ {df[value_col].mean():,.2f}",
                    f"R$ {df[value_col].median():,.2f}",
                    f"R$ {df[value_col].std():,.2f}",
                    f"R$ {df[value_col].min():,.2f}",
                    f"R$ {df[value_col].max():,.2f}",
                    f"R$ {df[value_col].quantile(0.25):,.2f}",
                    f"R$ {df[value_col].quantile(0.75):,.2f}",
                    f"{df[value_col].skew():.3f}",
                    f"{df[value_col].kurtosis():.3f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Histograma das vendas
            fig_hist = px.histogram(
                df,
                x=value_col,
                nbins=30,
                title="Distribuição das Vendas",
                labels={value_col: 'Vendas (R$)', 'count': 'Frequência'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Teste de estacionariedade
        st.subheader("🔍 Teste de Estacionariedade (Augmented Dickey-Fuller)")
        
        try:
            result = adfuller(df[value_col].dropna())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Estatística ADF", f"{result[0]:.4f}")
            
            with col2:
                st.metric("📈 p-valor", f"{result[1]:.4f}")
            
            with col3:
                is_stationary = result[1] < 0.05
                status = "Estacionária" if is_stationary else "Não-Estacionária"
                st.metric("✅ Status", status)
            
            if result[1] < 0.05:
                st.success("✅ **Série Estacionária**: A série não possui tendência ou sazonalidade forte.")
            else:
                st.warning("⚠️ **Série Não-Estacionária**: A série possui tendência ou sazonalidade. Considere diferenciação.")
            
        except Exception as e:
            st.error(f"❌ Erro no teste de estacionariedade: {e}")
        
        # Identificação de outliers
        self._identify_outliers(df, value_col)
    
    def _identify_outliers(self, df, value_col):
        """Identifica e visualiza outliers na série temporal"""
        
        st.subheader("🎯 Identificação de Outliers")
        
        # Método IQR
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Outliers", len(outliers))
        
        with col2:
            outlier_pct = (len(outliers) / len(df)) * 100
            st.metric("📈 % Outliers", f"{outlier_pct:.1f}%")
        
        with col3:
            if len(outliers) > 0:
                avg_deviation = abs(outliers[value_col] - df[value_col].median()).mean()
                st.metric("📊 Desvio Médio", f"R$ {avg_deviation:,.2f}")
        
        if len(outliers) > 0:
            # Visualizar outliers
            fig_outliers = go.Figure()
            
            fig_outliers.add_trace(go.Scatter(
                x=df['data'],
                y=df[value_col],
                mode='lines+markers',
                name='Vendas',
                marker=dict(color='blue', size=4),
                line=dict(color='blue', width=1)
            ))
            
            fig_outliers.add_trace(go.Scatter(
                x=outliers['data'],
                y=outliers[value_col],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig_outliers.update_layout(
                title="Identificação de Outliers",
                xaxis_title="Data",
                yaxis_title="Vendas (R$)",
                height=400
            )
            
            st.plotly_chart(fig_outliers, use_container_width=True)
            
            # Tabela dos outliers
            if len(outliers) <= 20:
                st.write("**Outliers Detectados:**")
                outliers_display = outliers[['data', value_col]].copy()
                outliers_display['data'] = outliers_display['data'].dt.strftime('%d/%m/%Y')
                outliers_display[value_col] = outliers_display[value_col].apply(lambda x: f"R$ {x:,.2f}")
                st.dataframe(outliers_display, use_container_width=True, hide_index=True)
            else:
                st.write(f"**{len(outliers)} outliers detectados** (exibindo apenas os primeiros 10)")
                outliers_display = outliers[['data', value_col]].head(10).copy()
                outliers_display['data'] = outliers_display['data'].dt.strftime('%d/%m/%Y')
                outliers_display[value_col] = outliers_display[value_col].apply(lambda x: f"R$ {x:,.2f}")
                st.dataframe(outliers_display, use_container_width=True, hide_index=True)
    
    def _render_decomposition(self, df, value_col):
        """Renderiza decomposição da série temporal"""
        
        st.subheader("🔍 Decomposição da Série Temporal")
        
        # Configurações da decomposição
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Tipo de Modelo:", ["additive", "multiplicative"])
        
        with col2:
            # Calcular período baseado na frequência dos dados
            period = st.slider("Período de Sazonalidade:", 7, 365, 30)
        
        try:
            # Realizar decomposição
            decomposition = seasonal_decompose(
                df[value_col].dropna(), 
                model=model_type, 
                period=period
            )
            
            # Criar subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Série Original', 'Tendência', 'Sazonalidade', 'Resíduos'],
                vertical_spacing=0.05
            )
            
            # Série original
            fig.add_trace(
                go.Scatter(x=df['data'], y=df[value_col], name='Original', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Tendência
            fig.add_trace(
                go.Scatter(x=df['data'], y=decomposition.trend, name='Tendência', line=dict(color='red')),
                row=2, col=1
            )
            
            # Sazonalidade
            fig.add_trace(
                go.Scatter(x=df['data'], y=decomposition.seasonal, name='Sazonalidade', line=dict(color='green')),
                row=3, col=1
            )
            
            # Resíduos
            fig.add_trace(
                go.Scatter(x=df['data'], y=decomposition.resid, name='Resíduos', line=dict(color='orange')),
                row=4, col=1
            )
            
            fig.update_layout(
                height=800,
                title_text=f"Decomposição {model_type.title()} da Série Temporal",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise dos componentes
            st.subheader("📊 Análise dos Componentes")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_strength = 1 - (decomposition.resid.var() / df[value_col].var())
                st.metric("📈 Força da Tendência", f"{trend_strength:.3f}")
            
            with col2:
                seasonal_strength = 1 - (decomposition.resid.var() / (df[value_col] - decomposition.trend).var())
                st.metric("🔄 Força da Sazonalidade", f"{seasonal_strength:.3f}")
            
            with col3:
                residual_var = decomposition.resid.var()
                st.metric("📊 Variância Resíduos", f"{residual_var:.2f}")
            
            # Interpretação
            interpretations = []
            
            if trend_strength > 0.6:
                interpretations.append("📈 **Tendência Forte**: A série apresenta uma tendência bem definida.")
            elif trend_strength > 0.3:
                interpretations.append("📊 **Tendência Moderada**: A série apresenta alguma tendência.")
            else:
                interpretations.append("➡️ **Tendência Fraca**: A série não apresenta tendência clara.")
            
            if seasonal_strength > 0.6:
                interpretations.append("🔄 **Sazonalidade Forte**: Padrões sazonais bem definidos.")
            elif seasonal_strength > 0.3:
                interpretations.append("📊 **Sazonalidade Moderada**: Alguns padrões sazonais presentes.")
            else:
                interpretations.append("➡️ **Sazonalidade Fraca**: Pouco padrão sazonal detectado.")
            
            for interpretation in interpretations:
                st.info(interpretation)
            
        except Exception as e:
            st.error(f"❌ Erro na decomposição: {e}")
            st.info("💡 Tente ajustar o período de sazonalidade ou verificar se há dados suficientes.")
    
    def _render_trend_analysis(self, df, value_col):
        """Renderiza análise de tendências"""
        
        st.subheader("📈 Análise de Tendências")
        
        # Análise de tendência linear
        from scipy.stats import linregress
        
        # Criar índice numérico
        x = np.arange(len(df))
        y = df[value_col].values
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Métricas da tendência
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Inclinação", f"{slope:.2f} R$/dia")
        
        with col2:
            st.metric("📊 R²", f"{r_value**2:.4f}")
        
        with col3:
            st.metric("📈 p-valor", f"{p_value:.4f}")
        
        with col4:
            trend_direction = "Crescente" if slope > 0 else "Decrescente" if slope < 0 else "Estável"
            st.metric("🎯 Direção", trend_direction)
        
        # Visualização da tendência
        trend_line = slope * x + intercept
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=df[value_col],
            mode='lines',
            name='Vendas',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=trend_line,
            mode='lines',
            name='Tendência Linear',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Análise de Tendência Linear",
            xaxis_title="Data",
            yaxis_title="Vendas (R$)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretação da tendência
        if p_value < 0.05:
            if slope > 0:
                st.success(f"✅ **Tendência Crescente Significativa**: As vendas aumentam em média R$ {slope:.2f} por dia.")
            else:
                st.warning(f"⚠️ **Tendência Decrescente Significativa**: As vendas diminuem em média R$ {abs(slope):.2f} por dia.")
        else:
            st.info("ℹ️ **Sem Tendência Significativa**: Não há evidência estatística de tendência linear.")
        
        # Análise de tendência por períodos
        st.subheader("📊 Tendência por Períodos")
        
        # Análise mensal
        monthly_trend = df.groupby(['ano', 'mes'])[value_col].mean().reset_index()
        monthly_trend['periodo'] = monthly_trend['ano'].astype(str) + '-' + monthly_trend['mes'].astype(str).str.zfill(2)
        
        fig_monthly = px.line(
            monthly_trend,
            x='periodo',
            y=value_col,
            title="Vendas Médias Mensais",
            labels={'periodo': 'Período (Ano-Mês)', value_col: 'Vendas Médias (R$)'}
        )
        fig_monthly.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Análise trimestral
        quarterly_trend = df.groupby(['ano', 'trimestre'])[value_col].mean().reset_index()
        quarterly_trend['periodo'] = quarterly_trend['ano'].astype(str) + '-Q' + quarterly_trend['trimestre'].astype(str)
        
        fig_quarterly = px.bar(
            quarterly_trend,
            x='periodo',
            y=value_col,
            title="Vendas Médias Trimestrais",
            labels={'periodo': 'Período', value_col: 'Vendas Médias (R$)'}
        )
        st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Detecção de pontos de mudança
        self._detect_change_points(df, value_col)
    
    def _detect_change_points(self, df, value_col):
        """Detecta pontos de mudança na série temporal"""
        
        st.subheader("🔄 Detecção de Pontos de Mudança")
        
        try:
            # Método simples de detecção de mudança usando diferenças
            window = min(30, len(df) // 10)
            
            if window < 5:
                st.warning("⚠️ Dados insuficientes para detecção de pontos de mudança")
                return
            
            # Calcular média móvel e desvio padrão
            rolling_mean = df[value_col].rolling(window=window).mean()
            rolling_std = df[value_col].rolling(window=window).std()
            
            # Identificar pontos onde a série sai do intervalo de confiança
            upper_bound = rolling_mean + 2 * rolling_std
            lower_bound = rolling_mean - 2 * rolling_std
            
            change_points = df[
                (df[value_col] > upper_bound) | 
                (df[value_col] < lower_bound)
            ]
            
            if len(change_points) > 0:
                st.write(f"**{len(change_points)} pontos de mudança detectados:**")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['data'],
                    y=df[value_col],
                    mode='lines',
                    name='Vendas',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['data'],
                    y=rolling_mean,
                    mode='lines',
                    name='Média Móvel',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['data'],
                    y=upper_bound,
                    mode='lines',
                    name='Limite Superior',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['data'],
                    y=lower_bound,
                    mode='lines',
                    name='Limite Inferior',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=change_points['data'],
                    y=change_points[value_col],
                    mode='markers',
                    name='Pontos de Mudança',
                    marker=dict(color='orange', size=8, symbol='diamond')
                ))
                
                fig.update_layout(
                    title="Detecção de Pontos de Mudança",
                    xaxis_title="Data",
                    yaxis_title="Vendas (R$)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Nenhum ponto de mudança significativo detectado")
                
        except Exception as e:
            st.error(f"❌ Erro na detecção de pontos de mudança: {e}")
    
    def _render_seasonality_analysis(self, df, value_col):
        """Renderiza análise de sazonalidade"""
        
        st.subheader("🔄 Análise de Sazonalidade")
        
        # Análise por dia da semana
        weekday_analysis = df.groupby('nome_dia')[value_col].agg(['mean', 'std', 'count']).round(2)
        weekday_analysis = weekday_analysis.reindex([
            'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📅 Vendas por Dia da Semana:**")
            st.dataframe(weekday_analysis, use_container_width=True)
        
        with col2:
            fig_weekday = px.bar(
                x=weekday_analysis.index,
                y=weekday_analysis['mean'],
                error_y=weekday_analysis['std'],
                title="Vendas Médias por Dia da Semana",
                labels={'x': 'Dia da Semana', 'y': 'Vendas Médias (R$)'}
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Análise por mês
        monthly_analysis = df.groupby('nome_mes')[value_col].agg(['mean', 'std', 'count']).round(2)
        month_order = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                      'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        available_months = [m for m in month_order if m in monthly_analysis.index]
        monthly_analysis = monthly_analysis.reindex(available_months)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📅 Vendas por Mês:**")
            st.dataframe(monthly_analysis, use_container_width=True)
        
        with col2:
            fig_monthly = px.line(
                x=monthly_analysis.index,
                y=monthly_analysis['mean'],
                title="Padrão Sazonal Mensal",
                labels={'x': 'Mês', 'y': 'Vendas Médias (R$)'}
            )
            fig_monthly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Análise de sazonalidade avançada
        st.subheader("📊 Análise de Sazonalidade Avançada")
        
        # Boxplot por trimestre
        fig_quarterly = px.box(
            df,
            x='trimestre',
            y=value_col,
            title="Distribuição de Vendas por Trimestre",
            labels={'trimestre': 'Trimestre', value_col: 'Vendas (R$)'}
        )
        st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Heatmap sazonal
        if len(df) > 100:  # Só fazer se houver dados suficientes
            
            # Criar pivot para heatmap
            df['semana_ano'] = df['data'].dt.isocalendar().week
            
            # Limitar a análise se houver muitos dados
            if len(df['semana_ano'].unique()) > 53:
                recent_data = df.tail(365)  # Último ano de dados
            else:
                recent_data = df
            
            pivot_data = recent_data.pivot_table(
                values=value_col,
                index='semana_ano',
                columns='dia_semana',
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                fig_heatmap = px.imshow(
                    pivot_data,
                    labels=dict(x="Dia da Semana", y="Semana do Ano", color="Vendas (R$)"),
                    title="Heatmap Sazonal: Vendas por Semana e Dia",
                    color_continuous_scale='viridis'
                )
                
                fig_heatmap.update_xaxes(
                    tickvals=list(range(7)),
                    ticktext=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Testes estatísticos de sazonalidade
        self._test_seasonality(df, value_col)
    
    def _test_seasonality(self, df, value_col):
        """Testa estatisticamente a presença de sazonalidade"""
        
        st.subheader("🔍 Testes de Sazonalidade")
        
        try:
            # Teste Kruskal-Wallis para diferenças por dia da semana
            groups = [df[df['dia_semana'] == day][value_col] for day in range(7)]
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) >= 2:
                kruskal_stat, kruskal_p = stats.kruskal(*groups)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📊 Kruskal-Wallis (Dias)", f"{kruskal_stat:.4f}")
                
                with col2:
                    st.metric("📈 p-valor", f"{kruskal_p:.4f}")
                
                if kruskal_p < 0.05:
                    st.success("✅ **Sazonalidade Semanal Significativa**: Há diferenças estatísticas entre os dias da semana.")
                else:
                    st.info("ℹ️ **Sem Sazonalidade Semanal**: Não há diferenças significativas entre os dias da semana.")
            
            # Teste para sazonalidade mensal (se houver dados suficientes)
            if len(df['mes'].unique()) >= 3:
                monthly_groups = [df[df['mes'] == month][value_col] for month in df['mes'].unique()]
                monthly_groups = [group for group in monthly_groups if len(group) > 0]
                
                if len(monthly_groups) >= 3:
                    monthly_kruskal_stat, monthly_kruskal_p = stats.kruskal(*monthly_groups)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("📊 Kruskal-Wallis (Meses)", f"{monthly_kruskal_stat:.4f}")
                    
                    with col2:
                        st.metric("📈 p-valor", f"{monthly_kruskal_p:.4f}")
                    
                    if monthly_kruskal_p < 0.05:
                        st.success("✅ **Sazonalidade Mensal Significativa**: Há diferenças estatísticas entre os meses.")
                    else:
                        st.info("ℹ️ **Sem Sazonalidade Mensal**: Não há diferenças significativas entre os meses.")
            
        except Exception as e:
            st.error(f"❌ Erro nos testes de sazonalidade: {e}")
    
    def _render_autocorrelation_analysis(self, df, value_col):
        """Renderiza análise de autocorrelação"""
        
        st.subheader("📉 Análise de Autocorrelação")
        
        try:
            # Calcular ACF e PACF
            max_lags = min(50, len(df) // 4)
            
            acf_values = acf(df[value_col].dropna(), nlags=max_lags, alpha=0.05)
            pacf_values = pacf(df[value_col].dropna(), nlags=max_lags, alpha=0.05)
            
            # Gráfico ACF
            col1, col2 = st.columns(2)
            
            with col1:
                lags = range(len(acf_values[0]))
                
                fig_acf = go.Figure()
                
                fig_acf.add_trace(go.Bar(
                    x=list(lags),
                    y=acf_values[0],
                    name='ACF',
                    marker_color='blue'
                ))
                
                # Adicionar bandas de confiança
                if len(acf_values) > 1:
                    fig_acf.add_trace(go.Scatter(
                        x=list(lags),
                        y=acf_values[1][:, 0],
                        mode='lines',
                        name='IC Inferior',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_acf.add_trace(go.Scatter(
                        x=list(lags),
                        y=acf_values[1][:, 1],
                        mode='lines',
                        name='IC Superior',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig_acf.update_layout(
                    title="Função de Autocorrelação (ACF)",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelação",
                    height=400
                )
                
                st.plotly_chart(fig_acf, use_container_width=True)
            
            with col2:
                lags = range(len(pacf_values[0]))
                
                fig_pacf = go.Figure()
                
                fig_pacf.add_trace(go.Bar(
                    x=list(lags),
                    y=pacf_values[0],
                    name='PACF',
                    marker_color='green'
                ))
                
                # Adicionar bandas de confiança
                if len(pacf_values) > 1:
                    fig_pacf.add_trace(go.Scatter(
                        x=list(lags),
                        y=pacf_values[1][:, 0],
                        mode='lines',
                        name='IC Inferior',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pacf.add_trace(go.Scatter(
                        x=list(lags),
                        y=pacf_values[1][:, 1],
                        mode='lines',
                        name='IC Superior',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig_pacf.update_layout(
                    title="Função de Autocorrelação Parcial (PACF)",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelação Parcial",
                    height=400
                )
                
                st.plotly_chart(fig_pacf, use_container_width=True)
            
            # Interpretação dos resultados
            st.subheader("💡 Interpretação da Autocorrelação")
            
            interpretations = []
            
            # Verificar autocorrelação significativa
            significant_acf_lags = []
            if len(acf_values) > 1:
                for i, (acf_val, (lower, upper)) in enumerate(zip(acf_values[0][1:], acf_values[1][1:]), 1):
                    if acf_val < lower or acf_val > upper:
                        significant_acf_lags.append(i)
            
            if significant_acf_lags:
                if 1 in significant_acf_lags:
                    interpretations.append("📈 **Autocorrelação de Lag 1**: Valor de hoje influencia o valor de amanhã.")
                
                if any(lag in [7, 14, 21, 28] for lag in significant_acf_lags):
                    interpretations.append("📅 **Padrão Semanal**: Autocorrelação em múltiplos de 7 dias detectada.")
                
                if any(lag >= 30 for lag in significant_acf_lags):
                    interpretations.append("📊 **Padrão de Longo Prazo**: Autocorrelação de longo prazo detectada.")
            
            if not interpretations:
                interpretations.append("➡️ **Baixa Autocorrelação**: A série apresenta pouca dependência temporal.")
            
            for interpretation in interpretations:
                st.info(interpretation)
            
            # Tabela com valores significativos
            if significant_acf_lags:
                st.write("**📊 Lags com Autocorrelação Significativa:**")
                
                significant_data = []
                for lag in significant_acf_lags[:10]:  # Limitar a 10 lags
                    significant_data.append({
                        'Lag': lag,
                        'ACF': f"{acf_values[0][lag]:.4f}",
                        'Significativo': '✅'
                    })
                
                sig_df = pd.DataFrame(significant_data)
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Erro na análise de autocorrelação: {e}")

# Função para integrar com streamlit_app.py
def show_serie_temporal_page(df, role, store_manager):
    """Função para mostrar a página de série temporal"""
    
    page = SerieTemporalPage(store_manager)
    page.render()