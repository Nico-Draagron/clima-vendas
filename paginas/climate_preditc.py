# ============================================================================
# 🌤️ pages/climate_prediction.py - PREVISÃO CLIMÁTICA
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.store_manager import StoreDataManager
import requests

class ClimatePredictionPage:
    """Página de previsão climática e análise meteorológica"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
    
    def render(self):
        """Renderiza página de previsão climática"""
        
        st.markdown("# 🌤️ Previsão Climática")
        st.markdown("**Sistema de análise e previsão meteorológica para tomada de decisão**")
        
        # Tabs principais
        tab1, tab2, tab3 = st.tabs([
            "📊 Análise Histórica", 
            "🔮 Previsões", 
            "🎯 Impacto nas Vendas"
        ])
        
        with tab1:
            self._render_historical_analysis()
        
        with tab2:
            self._render_predictions()
        
        with tab3:
            self._render_sales_impact()
    
    def _render_historical_analysis(self):
        """Análise histórica do clima"""
        
        st.subheader("📈 Análise Histórica do Clima")
        
        # Carregar dados climáticos
        df_climate = self.store_manager.load_climate_data()
        
        if df_climate is None:
            st.error("❌ Não foi possível carregar dados climáticos")
            return
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input(
                "📅 Data Inicial",
                value=df_climate['data'].min().date(),
                min_value=df_climate['data'].min().date(),
                max_value=df_climate['data'].max().date()
            )
        
        with col2:
            end_date = st.date_input(
                "📅 Data Final",
                value=df_climate['data'].max().date(),
                min_value=df_climate['data'].min().date(),
                max_value=df_climate['data'].max().date()
            )
        
        with col3:
            variable = st.selectbox(
                "🌡️ Variável Climática",
                ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana'],
                format_func=lambda x: {
                    'temp_media': 'Temperatura Média',
                    'temp_max': 'Temperatura Máxima',
                    'temp_min': 'Temperatura Mínima',
                    'precipitacao_total': 'Precipitação Total',
                    'umid_mediana': 'Umidade Mediana'
                }.get(x, x)
            )
        
        # Filtrar dados
        df_filtered = df_climate[
            (df_climate['data'] >= pd.to_datetime(start_date)) &
            (df_climate['data'] <= pd.to_datetime(end_date))
        ]
        
        if df_filtered.empty:
            st.warning("⚠️ Nenhum dado encontrado para o período selecionado")
            return
        
        # Métricas do período
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if variable in df_filtered.columns:
                avg_value = df_filtered[variable].mean()
                unit = '°C' if 'temp' in variable else 'mm' if 'precipitacao' in variable else '%'
                st.metric(f"📊 Média", f"{avg_value:.1f}{unit}")
        
        with col2:
            if variable in df_filtered.columns:
                max_value = df_filtered[variable].max()
                st.metric(f"📈 Máximo", f"{max_value:.1f}{unit}")
        
        with col3:
            if variable in df_filtered.columns:
                min_value = df_filtered[variable].min()
                st.metric(f"📉 Mínimo", f"{min_value:.1f}{unit}")
        
        with col4:
            days_count = len(df_filtered)
            st.metric("📅 Dias", f"{days_count}")
        
        # Gráfico de série temporal
        if variable in df_filtered.columns:
            fig = px.line(
                df_filtered,
                x='data',
                y=variable,
                title=f"Evolução de {variable.replace('_', ' ').title()} ao Longo do Tempo"
            )
            
            fig.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise de sazonalidade
        st.subheader("🔄 Análise de Sazonalidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Por mês
            if variable in df_filtered.columns:
                df_month = df_filtered.copy()
                df_month['mes'] = df_month['data'].dt.month
                monthly_avg = df_month.groupby('mes')[variable].mean().reset_index()
                
                month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                              7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
                monthly_avg['mes_nome'] = monthly_avg['mes'].map(month_names)
                
                fig_month = px.bar(
                    monthly_avg,
                    x='mes_nome',
                    y=variable,
                    title=f"Média Mensal - {variable.replace('_', ' ').title()}"
                )
                
                st.plotly_chart(fig_month, use_container_width=True)
        
        with col2:
            # Por dia da semana
            if variable in df_filtered.columns:
                df_weekday = df_filtered.copy()
                df_weekday['dia_semana'] = df_weekday['data'].dt.day_name()
                weekday_avg = df_weekday.groupby('dia_semana')[variable].mean().reset_index()
                
                fig_weekday = px.bar(
                    weekday_avg,
                    x='dia_semana',
                    y=variable,
                    title=f"Média por Dia da Semana - {variable.replace('_', ' ').title()}"
                )
                
                fig_weekday.update_xaxis(tickangle=45)
                st.plotly_chart(fig_weekday, use_container_width=True)
    
    def _render_predictions(self):
        """Interface de previsões"""
        
        st.subheader("🔮 Previsões Meteorológicas")
        
        # Nota sobre API
        st.info("""
        📡 **Integração com APIs Meteorológicas**
        
        Esta seção será integrada com:
        - **INMET** (Instituto Nacional de Meteorologia)
        - **OpenWeatherMap**
        - **Outros provedores de dados meteorológicos**
        """)
        
        # Configurações de previsão
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_days = st.selectbox(
                "📅 Período de Previsão",
                [1, 3, 7, 15],
                format_func=lambda x: f"{x} dia{'s' if x > 1 else ''}"
            )
        
        with col2:
            location = st.text_input(
                "📍 Localização",
                value="Agudo, RS",
                help="Digite a cidade para previsão"
            )
        
        with col3:
            if st.button("🔍 Buscar Previsão", type="primary"):
                with st.spinner("Buscando previsão..."):
                    # Simular dados de previsão
                    self._simulate_weather_prediction(prediction_days, location)
    
    def _simulate_weather_prediction(self, days, location):
        """Simula dados de previsão meteorológica"""
        
        st.subheader(f"🌤️ Previsão para {location} - Próximos {days} dias")
        
        # Gerar dados simulados
        dates = [datetime.now().date() + timedelta(days=i) for i in range(days)]
        
        # Simulação baseada em padrões sazonais
        base_temp = 25 + np.random.normal(0, 3, days)
        humidity = 60 + np.random.normal(0, 15, days)
        precipitation = np.random.exponential(2, days)
        
        prediction_data = {
            'Data': dates,
            'Temperatura Mín. (°C)': base_temp - 5,
            'Temperatura Máx. (°C)': base_temp + 5,
            'Umidade (%)': np.clip(humidity, 30, 95),
            'Precipitação (mm)': precipitation,
            'Condição': ['Ensolarado' if p < 1 else 'Parcialmente Nublado' if p < 5 else 'Chuvoso' 
                        for p in precipitation]
        }
        
        df_prediction = pd.DataFrame(prediction_data)
        
        # Exibir tabela de previsão
        st.dataframe(df_prediction, use_container_width=True)
        
        # Gráfico de previsão
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_prediction['Data'],
            y=df_prediction['Temperatura Máx. (°C)'],
            mode='lines+markers',
            name='Temp. Máxima',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_prediction['Data'],
            y=df_prediction['Temperatura Mín. (°C)'],
            mode='lines+markers',
            name='Temp. Mínima',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Previsão de Temperatura",
            xaxis_title="Data",
            yaxis_title="Temperatura (°C)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alertas meteorológicos
        max_rain = df_prediction['Precipitação (mm)'].max()
        max_temp = df_prediction['Temperatura Máx. (°C)'].max()
        
        if max_rain > 10:
            st.warning(f"⚠️ **Alerta de Chuva**: Precipitação de até {max_rain:.1f}mm prevista")
        
        if max_temp > 35:
            st.error(f"🌡️ **Alerta de Calor**: Temperatura de até {max_temp:.1f}°C prevista")
    
    def _render_sales_impact(self):
        """Análise do impacto do clima nas vendas"""
        
        st.subheader("🎯 Impacto do Clima nas Vendas")
        
        # Carregar dados das lojas
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.warning("⚠️ Nenhuma loja configurada")
            return
        
        # Seleção de loja
        store_options = {f"{info['display_name']} ({store_id})": store_id 
                        for store_id, info in stores.items()}
        
        selected_display = st.selectbox(
            "🏪 Escolha uma loja para análise:",
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
            st.error(f"❌ Coluna de vendas '{value_col}' não encontrada")
            return
        
        # Análise de correlação
        st.subheader("📊 Correlações Clima vs Vendas")
        
        climate_vars = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana']
        correlations = {}
        
        for var in climate_vars:
            if var in df.columns:
                corr = df[value_col].corr(df[var])
                correlations[var] = corr
        
        if correlations:
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Variável', 'Correlação'])
            corr_df['Correlação_abs'] = corr_df['Correlação'].abs()
            corr_df = corr_df.sort_values('Correlação_abs', ascending=False)
            
            # Gráfico de correlações
            fig = px.bar(
                corr_df,
                x='Variável',
                y='Correlação',
                title="Correlação entre Variáveis Climáticas e Vendas",
                color='Correlação',
                color_continuous_scale='RdBu_r'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretação
            st.subheader("🔍 Interpretação das Correlações")
            
            for _, row in corr_df.head(3).iterrows():
                var_name = {
                    'temp_media': 'Temperatura Média',
                    'temp_max': 'Temperatura Máxima', 
                    'temp_min': 'Temperatura Mínima',
                    'precipitacao_total': 'Precipitação',
                    'umid_mediana': 'Umidade'
                }.get(row['Variável'], row['Variável'])
                
                corr_value = row['Correlação']
                
                if abs(corr_value) > 0.3:
                    strength = "forte" if abs(corr_value) > 0.5 else "moderada"
                    direction = "positiva" if corr_value > 0 else "negativa"
                    
                    st.write(f"**{var_name}**: Correlação {strength} {direction} ({corr_value:.3f})")
                else:
                    st.write(f"**{var_name}**: Correlação fraca ({corr_value:.3f})")
        
        # Análise por categorias climáticas
        st.subheader("🌤️ Vendas por Condições Climáticas")
        
        if 'precipitacao_total' in df.columns and 'temp_media' in df.columns:
            # Categorizar dias
            df_analysis = df.copy()
            df_analysis['categoria_chuva'] = pd.cut(
                df_analysis['precipitacao_total'],
                bins=[-0.1, 0, 5, 20, float('inf')],
                labels=['Sem chuva', 'Chuva leve', 'Chuva moderada', 'Chuva intensa']
            )
            
            df_analysis['categoria_temp'] = pd.cut(
                df_analysis['temp_media'],
                bins=[0, 18, 25, 30, 50],
                labels=['Frio', 'Ameno', 'Quente', 'Muito quente']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vendas por categoria de chuva
                rain_stats = df_analysis.groupby('categoria_chuva')[value_col].agg(['mean', 'count']).round(2)
                rain_stats.columns = ['Vendas Médias', 'Número de Dias']
                
                st.write("**Vendas por Intensidade de Chuva:**")
                st.dataframe(rain_stats, use_container_width=True)
            
            with col2:
                # Vendas por categoria de temperatura
                temp_stats = df_analysis.groupby('categoria_temp')[value_col].agg(['mean', 'count']).round(2)
                temp_stats.columns = ['Vendas Médias', 'Número de Dias']
                
                st.write("**Vendas por Faixa de Temperatura:**")
                st.dataframe(temp_stats, use_container_width=True)