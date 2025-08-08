# ============================================================================
# 🔮 pages/previsao_climatica.py - PREVISÃO CLIMÁTICA
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class PrevisaoClimaticaPage:
    """Página completa de previsão climática"""
    
    def __init__(self, store_manager):
        self.store_manager = store_manager
        # API Keys para serviços meteorológicos
        self.openweather_api_key = None  # Seria configurável em produção
        self.weather_apis_available = False
        
    def render(self):
        """Renderiza página principal de previsão climática"""
        
        st.markdown("# 🔮 Previsão Climática")
        st.markdown("**Sistema de previsão meteorológica para suporte à tomada de decisões comerciais**")
        
        # Informação sobre APIs
        if not self.weather_apis_available:
            st.info("ℹ️ **Demo Mode**: Usando dados simulados. Em produção, integraria com APIs meteorológicas (OpenWeatherMap, WeatherAPI, etc.)")
        
        # Carregar dados históricos climáticos
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.error("❌ Nenhuma loja configurada. Configure uma loja no painel administrativo.")
            return
        
        # Seleção de loja (para contexto geográfico)
        store_options = {f"{info['display_name']} ({store_id})": store_id 
                        for store_id, info in stores.items()}
        
        selected_display = st.selectbox(
            "🏪 Escolha uma loja para análise climática:",
            options=list(store_options.keys())
        )
        
        selected_store_id = store_options[selected_display]
        
        # Carregar dados históricos
        df = self.store_manager.load_store_data(selected_store_id)
        store_info = stores[selected_store_id]
        
        if df is None or df.empty:
            st.error("❌ Não foi possível carregar dados da loja")
            return
        
        # Verificar dados climáticos
        climate_cols = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana']
        available_climate = [col for col in climate_cols if col in df.columns]
        
        if not available_climate:
            st.error("❌ Nenhuma variável climática encontrada nos dados históricos")
            return
        
        # Preparar dados
        df = self._prepare_climate_data(df)
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌤️ Condições Atuais",
            "📅 Previsão 7 Dias",
            "📊 Análise Histórica",
            "🚨 Alertas Climáticos",
            "💼 Impacto nos Negócios"
        ])
        
        with tab1:
            self._render_current_conditions(df, store_info['display_name'])
        
        with tab2:
            self._render_7day_forecast(df, store_info['display_name'])
        
        with tab3:
            self._render_historical_analysis(df, available_climate)
        
        with tab4:
            self._render_weather_alerts(df)
        
        with tab5:
            self._render_business_impact(df, store_info)
    
    def _prepare_climate_data(self, df):
        """Prepara dados climáticos"""
        
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values('data')
        
        # Adicionar features temporais
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia'] = df['data'].dt.day
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_ano'] = df['data'].dt.dayofyear
        
        # Adicionar categorias climáticas
        if 'temp_media' in df.columns:
            df['categoria_temp'] = pd.cut(
                df['temp_media'],
                bins=[0, 15, 20, 25, 30, 50],
                labels=['Muito Frio', 'Frio', 'Ameno', 'Quente', 'Muito Quente']
            )
        
        if 'precipitacao_total' in df.columns:
            df['categoria_chuva'] = pd.cut(
                df['precipitacao_total'],
                bins=[-0.1, 0, 2, 10, 25, float('inf')],
                labels=['Sem Chuva', 'Garoa', 'Chuva Leve', 'Chuva Moderada', 'Chuva Intensa']
            )
        
        return df
    
    def _render_current_conditions(self, df, store_name):
        """Renderiza condições climáticas atuais"""
        
        st.subheader(f"🌤️ Condições Atuais - {store_name}")
        
        # Simular condições atuais baseadas nos dados mais recentes
        latest_data = df.iloc[-1] if not df.empty else None
        
        if latest_data is None:
            st.error("❌ Não há dados climáticos disponíveis")
            return
        
        # Gerar condições "atuais" com pequena variação dos dados mais recentes
        current_conditions = self._generate_current_conditions(latest_data)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_atual = current_conditions['temperatura']
            temp_trend = np.random.choice(['↗️', '➡️', '↘️'], p=[0.3, 0.4, 0.3])
            st.metric("🌡️ Temperatura", f"{temp_atual:.1f}°C", temp_trend)
        
        with col2:
            umid_atual = current_conditions['umidade']
            st.metric("💧 Umidade", f"{umid_atual:.0f}%")
        
        with col3:
            precip_atual = current_conditions['precipitacao']
            st.metric("🌧️ Precipitação", f"{precip_atual:.1f}mm")
        
        with col4:
            vento_atual = current_conditions['vento']
            st.metric("💨 Vento", f"{vento_atual:.1f} km/h")
        
        # Condições detalhadas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Condições Detalhadas")
            
            # Status atual
            status_temp = self._get_temperature_status(temp_atual)
            status_chuva = self._get_precipitation_status(precip_atual)
            
            st.write(f"🌡️ **Temperatura**: {status_temp}")
            st.write(f"🌧️ **Precipitação**: {status_chuva}")
            st.write(f"💧 **Umidade Relativa**: {umid_atual:.0f}%")
            st.write(f"💨 **Velocidade do Vento**: {vento_atual:.1f} km/h")
            
            # Sensação térmica simulada
            sensacao_termica = temp_atual + np.random.normal(0, 2)
            st.write(f"🌡️ **Sensação Térmica**: {sensacao_termica:.1f}°C")
            
            # Índice UV simulado
            uv_index = max(0, min(11, 6 + np.random.normal(0, 2)))
            st.write(f"☀️ **Índice UV**: {uv_index:.1f}")
        
        with col2:
            # Gráfico gauge da temperatura
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=temp_atual,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Temperatura (°C)"},
                delta={'reference': df['temp_media'].mean() if 'temp_media' in df.columns else 25},
                gauge={'axis': {'range': [-10, 45]},
                      'bar': {'color': "darkblue"},
                      'steps': [
                          {'range': [-10, 0], 'color': "lightblue"},
                          {'range': [0, 15], 'color': "cyan"},
                          {'range': [15, 25], 'color': "yellow"},
                          {'range': [25, 35], 'color': "orange"},
                          {'range': [35, 45], 'color': "red"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 35}}))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Histórico recente (últimos 7 dias)
        st.subheader("📈 Histórico Recente (7 dias)")
        
        recent_data = df.tail(7)
        
        if len(recent_data) > 0:
            fig_recent = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Temperatura', 'Precipitação'],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Temperatura
            if 'temp_max' in df.columns and 'temp_min' in df.columns:
                fig_recent.add_trace(
                    go.Scatter(x=recent_data['data'], y=recent_data['temp_max'], 
                             name='Temp Máx', line=dict(color='red')),
                    row=1, col=1
                )
                fig_recent.add_trace(
                    go.Scatter(x=recent_data['data'], y=recent_data['temp_min'],
                             name='Temp Mín', line=dict(color='blue')),
                    row=1, col=1
                )
            elif 'temp_media' in df.columns:
                fig_recent.add_trace(
                    go.Scatter(x=recent_data['data'], y=recent_data['temp_media'],
                             name='Temp Média', line=dict(color='orange')),
                    row=1, col=1
                )
            
            # Precipitação
            if 'precipitacao_total' in df.columns:
                fig_recent.add_trace(
                    go.Bar(x=recent_data['data'], y=recent_data['precipitacao_total'],
                          name='Precipitação', marker_color='lightblue'),
                    row=2, col=1
                )
            
            fig_recent.update_layout(height=500, title_text="Condições dos Últimos 7 Dias")
            fig_recent.update_xaxes(title_text="Data", row=2, col=1)
            fig_recent.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
            fig_recent.update_yaxes(title_text="Precipitação (mm)", row=2, col=1)
            
            st.plotly_chart(fig_recent, use_container_width=True)
    
    def _generate_current_conditions(self, latest_data):
        """Gera condições climáticas atuais simuladas"""
        
        # Basear nas condições mais recentes com pequenas variações
        current = {}
        
        if 'temp_media' in latest_data:
            current['temperatura'] = latest_data['temp_media'] + np.random.normal(0, 2)
        else:
            current['temperatura'] = 25.0  # Default
        
        if 'umid_mediana' in latest_data:
            current['umidade'] = max(0, min(100, latest_data['umid_mediana'] + np.random.normal(0, 5)))
        else:
            current['umidade'] = 70.0  # Default
        
        if 'precipitacao_total' in latest_data:
            # Precipitação com maior probabilidade de zero
            if np.random.random() < 0.7:  # 70% chance de não chover
                current['precipitacao'] = 0.0
            else:
                current['precipitacao'] = max(0, latest_data['precipitacao_total'] * np.random.uniform(0.5, 2.0))
        else:
            current['precipitacao'] = 0.0
        
        # Vento simulado
        current['vento'] = max(0, np.random.normal(10, 5))
        
        return current
    
    def _get_temperature_status(self, temp):
        """Retorna status da temperatura"""
        if temp < 10:
            return "Muito Frio"
        elif temp < 18:
            return "Frio"
        elif temp < 25:
            return "Ameno"
        elif temp < 30:
            return "Quente"
        else:
            return "Muito Quente"
    
    def _get_precipitation_status(self, precip):
        """Retorna status da precipitação"""
        if precip == 0:
            return "Sem Chuva"
        elif precip < 2:
            return "Garoa"
        elif precip < 10:
            return "Chuva Leve"
        elif precip < 25:
            return "Chuva Moderada"
        else:
            return "Chuva Intensa"
    
    def _render_7day_forecast(self, df, store_name):
        """Renderiza previsão de 7 dias"""
        
        st.subheader(f"📅 Previsão 7 Dias - {store_name}")
        
        # Gerar previsão simulada baseada nos padrões históricos
        forecast_data = self._generate_7day_forecast(df)
        
        # Métricas resumo da previsão
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_max = forecast_data['temp_max'].max()
            st.metric("🌡️ Temp Máxima", f"{temp_max:.1f}°C")
        
        with col2:
            temp_min = forecast_data['temp_min'].min()
            st.metric("🌡️ Temp Mínima", f"{temp_min:.1f}°C")
        
        with col3:
            total_rain = forecast_data['precipitacao'].sum()
            st.metric("🌧️ Chuva Total", f"{total_rain:.1f}mm")
        
        with col4:
            rainy_days = (forecast_data['precipitacao'] > 0).sum()
            st.metric("☔ Dias de Chuva", f"{rainy_days}/7")
        
        # Tabela da previsão
        st.subheader("📋 Previsão Detalhada")
        
        forecast_display = forecast_data.copy()
        forecast_display['Data'] = forecast_display['data'].dt.strftime('%d/%m (%a)')
        forecast_display['Temp Mín/Máx'] = forecast_display.apply(
            lambda x: f"{x['temp_min']:.0f}°C / {x['temp_max']:.0f}°C", axis=1
        )
        forecast_display['Chuva'] = forecast_display['precipitacao'].apply(
            lambda x: f"{x:.1f}mm" if x > 0 else "Sem chuva"
        )
        forecast_display['Condição'] = forecast_display.apply(
            lambda x: self._get_weather_condition(x['temp_max'], x['precipitacao']), axis=1
        )
        
        display_cols = ['Data', 'Temp Mín/Máx', 'Chuva', 'Condição']
        st.dataframe(
            forecast_display[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Gráficos da previsão
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de temperatura
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=forecast_data['data'],
                y=forecast_data['temp_max'],
                mode='lines+markers',
                name='Máxima',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=forecast_data['data'],
                y=forecast_data['temp_min'],
                mode='lines+markers',
                name='Mínima',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)'
            ))
            
            fig_temp.update_layout(
                title="Previsão de Temperatura",
                xaxis_title="Data",
                yaxis_title="Temperatura (°C)",
                height=400
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Gráfico de precipitação
            fig_rain = px.bar(
                forecast_data,
                x='data',
                y='precipitacao',
                title="Previsão de Precipitação",
                labels={'data': 'Data', 'precipitacao': 'Precipitação (mm)'},
                color='precipitacao',
                color_continuous_scale='blues'
            )
            fig_rain.update_layout(height=400)
            st.plotly_chart(fig_rain, use_container_width=True)
        
        # Alertas para os próximos dias
        self._render_forecast_alerts(forecast_data)
    
    def _generate_7day_forecast(self, df):
        """Gera previsão de 7 dias baseada nos padrões históricos"""
        
        # Data base para previsão
        last_date = df['data'].max() if not df.empty else datetime.now()
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=7, freq='D')
        
        forecast_data = []
        
        for i, date in enumerate(forecast_dates):
            # Sazonalidade baseada no mês e dia do ano
            day_of_year = date.dayofyear
            
            # Padrão sazonal simulado (senoidal)
            seasonal_temp = 25 + 8 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
            
            # Variabilidade diária
            temp_base = seasonal_temp + np.random.normal(0, 3)
            temp_max = temp_base + np.random.uniform(3, 8)
            temp_min = temp_base - np.random.uniform(2, 6)
            
            # Precipitação com padrões realistas
            if np.random.random() < 0.3:  # 30% chance de chuva
                precipitacao = np.random.exponential(5)
            else:
                precipitacao = 0.0
            
            # Umidade correlacionada com chuva
            if precipitacao > 0:
                umidade = np.random.uniform(70, 95)
            else:
                umidade = np.random.uniform(40, 80)
            
            forecast_data.append({
                'data': date,
                'temp_max': temp_max,
                'temp_min': temp_min,
                'precipitacao': precipitacao,
                'umidade': umidade,
                'dia_semana': date.weekday()
            })
        
        return pd.DataFrame(forecast_data)
    
    def _get_weather_condition(self, temp_max, precipitacao):
        """Determina condição climática baseada na temperatura e chuva"""
        
        if precipitacao > 10:
            return "🌧️ Chuvoso"
        elif precipitacao > 0:
            return "🌦️ Parcialmente Chuvoso"
        elif temp_max > 30:
            return "☀️ Ensolarado e Quente"
        elif temp_max > 25:
            return "🌤️ Ensolarado"
        elif temp_max > 20:
            return "⛅ Parcialmente Nublado"
        else:
            return "☁️ Nublado e Frio"
    
    def _render_forecast_alerts(self, forecast_data):
        """Renderiza alertas baseados na previsão"""
        
        st.subheader("🚨 Alertas Meteorológicos")
        
        alerts = []
        
        # Verificar alertas de temperatura
        extreme_heat = forecast_data['temp_max'] > 35
        if extreme_heat.any():
            hot_days = extreme_heat.sum()
            alerts.append({
                'type': 'error',
                'title': '🌡️ Alerta de Calor Extremo',
                'message': f'{hot_days} dia(s) com temperatura acima de 35°C prevista'
            })
        
        extreme_cold = forecast_data['temp_min'] < 5
        if extreme_cold.any():
            cold_days = extreme_cold.sum()
            alerts.append({
                'type': 'warning',
                'title': '🧊 Alerta de Frio Intenso',
                'message': f'{cold_days} dia(s) com temperatura abaixo de 5°C prevista'
            })
        
        # Verificar alertas de precipitação
        heavy_rain = forecast_data['precipitacao'] > 25
        if heavy_rain.any():
            rainy_days = heavy_rain.sum()
            total_rain = forecast_data.loc[heavy_rain, 'precipitacao'].sum()
            alerts.append({
                'type': 'warning',
                'title': '🌧️ Alerta de Chuva Intensa',
                'message': f'{rainy_days} dia(s) com chuva intensa ({total_rain:.1f}mm total)'
            })
        
        # Verificar períodos prolongados de chuva
        consecutive_rain = 0
        max_consecutive = 0
        
        for _, row in forecast_data.iterrows():
            if row['precipitacao'] > 0:
                consecutive_rain += 1
                max_consecutive = max(max_consecutive, consecutive_rain)
            else:
                consecutive_rain = 0
        
        if max_consecutive >= 3:
            alerts.append({
                'type': 'info',
                'title': '☔ Período Chuvoso Prolongado',
                'message': f'Até {max_consecutive} dias consecutivos de chuva previstos'
            })
        
        # Exibir alertas
        if alerts:
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(f"**{alert['title']}**: {alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"**{alert['title']}**: {alert['message']}")
                else:
                    st.info(f"**{alert['title']}**: {alert['message']}")
        else:
            st.success("✅ **Condições Normais**: Nenhum alerta meteorológico para os próximos 7 dias")
    
    def _render_historical_analysis(self, df, available_climate):
        """Renderiza análise histórica do clima"""
        
        st.subheader("📊 Análise Histórica do Clima")
        
        # Estatísticas climáticas por mês
        if 'mes' in df.columns:
            st.subheader("📅 Padrões Sazonais")
            
            monthly_stats = []
            for mes in range(1, 13):
                month_data = df[df['mes'] == mes]
                if not month_data.empty:
                    stats = {
                        'Mês': mes,
                        'Nome': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                                'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][mes-1]
                    }
                    
                    if 'temp_media' in available_climate:
                        stats['Temp Média (°C)'] = month_data['temp_media'].mean()
                    
                    if 'precipitacao_total' in available_climate:
                        stats['Chuva Média (mm)'] = month_data['precipitacao_total'].mean()
                        stats['Dias Chuvosos'] = (month_data['precipitacao_total'] > 0).sum()
                    
                    if 'umid_mediana' in available_climate:
                        stats['Umidade (%)'] = month_data['umid_mediana'].mean()
                    
                    monthly_stats.append(stats)
            
            if monthly_stats:
                monthly_df = pd.DataFrame(monthly_stats)
                
                # Exibir tabela
                st.dataframe(
                    monthly_df.round(1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Gráficos sazonais
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Temp Média (°C)' in monthly_df.columns:
                        fig_temp_season = px.line(
                            monthly_df,
                            x='Nome',
                            y='Temp Média (°C)',
                            title="Temperatura Média por Mês",
                            markers=True
                        )
                        st.plotly_chart(fig_temp_season, use_container_width=True)
                
                with col2:
                    if 'Chuva Média (mm)' in monthly_df.columns:
                        fig_rain_season = px.bar(
                            monthly_df,
                            x='Nome',
                            y='Chuva Média (mm)',
                            title="Precipitação Média por Mês"
                        )
                        st.plotly_chart(fig_rain_season, use_container_width=True)
        
        # Eventos extremos históricos
        st.subheader("⚡ Eventos Climáticos Extremos")
        
        extreme_events = []
        
        if 'temp_max' in available_climate:
            hottest_day = df.loc[df['temp_max'].idxmax()]
            extreme_events.append({
                'Evento': '🌡️ Dia Mais Quente',
                'Data': hottest_day['data'].strftime('%d/%m/%Y'),
                'Valor': f"{hottest_day['temp_max']:.1f}°C"
            })
        
        if 'temp_min' in available_climate:
            coldest_day = df.loc[df['temp_min'].idxmin()]
            extreme_events.append({
                'Evento': '🧊 Dia Mais Frio',
                'Data': coldest_day['data'].strftime('%d/%m/%Y'),
                'Valor': f"{coldest_day['temp_min']:.1f}°C"
            })
        
        if 'precipitacao_total' in available_climate:
            rainiest_day = df.loc[df['precipitacao_total'].idxmax()]
            extreme_events.append({
                'Evento': '🌧️ Dia Mais Chuvoso',
                'Data': rainiest_day['data'].strftime('%d/%m/%Y'),
                'Valor': f"{rainiest_day['precipitacao_total']:.1f}mm"
            })
            
            # Período mais seco
            dry_periods = df['precipitacao_total'] == 0
            if dry_periods.any():
                max_dry_streak = 0
                current_streak = 0
                
                for has_rain in df['precipitacao_total'] > 0:
                    if not has_rain:
                        current_streak += 1
                        max_dry_streak = max(max_dry_streak, current_streak)
                    else:
                        current_streak = 0
                
                extreme_events.append({
                    'Evento': '☀️ Maior Período Seco',
                    'Data': 'Histórico',
                    'Valor': f"{max_dry_streak} dias consecutivos"
                })
        
        if extreme_events:
            extremes_df = pd.DataFrame(extreme_events)
            st.dataframe(extremes_df, use_container_width=True, hide_index=True)
        
        # Tendências climáticas
        self._analyze_climate_trends(df, available_climate)
    
    def _analyze_climate_trends(self, df, available_climate):
        """Analisa tendências climáticas"""
        
        st.subheader("📈 Tendências Climáticas")
        
        if 'ano' not in df.columns or len(df['ano'].unique()) < 2:
            st.info("ℹ️ Dados insuficientes para análise de tendências (necessário pelo menos 2 anos)")
            return
        
        # Análise anual
        yearly_stats = df.groupby('ano').agg({
            col: 'mean' for col in available_climate if col in df.columns
        }).round(2)
        
        if not yearly_stats.empty:
            st.write("**Médias Anuais:**")
            st.dataframe(yearly_stats, use_container_width=True)
            
            # Gráfico de tendências
            fig_trends = make_subplots(
                rows=len(available_climate), cols=1,
                subplot_titles=[col.replace('_', ' ').title() for col in available_climate]
            )
            
            for i, col in enumerate(available_climate, 1):
                if col in yearly_stats.columns:
                    fig_trends.add_trace(
                        go.Scatter(
                            x=yearly_stats.index,
                            y=yearly_stats[col],
                            mode='lines+markers',
                            name=col.replace('_', ' ').title(),
                            line=dict(width=3)
                        ),
                        row=i, col=1
                    )
            
            fig_trends.update_layout(
                height=200 * len(available_climate),
                title_text="Tendências Climáticas Anuais",
                showlegend=False
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Análise de correlação entre anos
            from scipy.stats import pearsonr
            
            trend_analysis = []
            
            for col in available_climate:
                if col in yearly_stats.columns and len(yearly_stats) > 2:
                    years = yearly_stats.index.values
                    values = yearly_stats[col].values
                    
                    # Calcular correlação com o tempo
                    corr, p_value = pearsonr(years, values)
                    
                    if p_value < 0.05:
                        trend_direction = "Crescente" if corr > 0 else "Decrescente"
                        significance = "Significativa"
                    else:
                        trend_direction = "Estável"
                        significance = "Não significativa"
                    
                    trend_analysis.append({
                        'Variável': col.replace('_', ' ').title(),
                        'Tendência': trend_direction,
                        'Correlação': f"{corr:.3f}",
                        'Significância': significance
                    })
            
            if trend_analysis:
                st.write("**Análise de Tendências:**")
                trends_df = pd.DataFrame(trend_analysis)
                st.dataframe(trends_df, use_container_width=True, hide_index=True)
    
    def _render_weather_alerts(self, df):
        """Renderiza alertas climáticos configuráveis"""
        
        st.subheader("🚨 Sistema de Alertas Climáticos")
        
        # Configuração de alertas
        st.subheader("⚙️ Configurar Alertas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🌡️ Alertas de Temperatura:**")
            
            temp_alta = st.number_input("Temperatura máxima (°C)", value=32.0, min_value=25.0, max_value=50.0)
            temp_baixa = st.number_input("Temperatura mínima (°C)", value=10.0, min_value=-10.0, max_value=20.0)
            
            enable_temp_alerts = st.checkbox("Habilitar alertas de temperatura", value=True)
        
        with col2:
            st.write("**🌧️ Alertas de Precipitação:**")
            
            chuva_intensa = st.number_input("Chuva intensa (mm)", value=20.0, min_value=10.0, max_value=100.0)
            dias_secos = st.number_input("Dias consecutivos sem chuva", value=7, min_value=3, max_value=30)
            
            enable_rain_alerts = st.checkbox("Habilitar alertas de chuva", value=True)
        
        # Verificar alertas nos dados históricos
        if st.button("🔍 Verificar Alertas Históricos"):
            
            historical_alerts = []
            
            if enable_temp_alerts and 'temp_max' in df.columns:
                hot_days = df[df['temp_max'] > temp_alta]
                if not hot_days.empty:
                    historical_alerts.append({
                        'Tipo': '🌡️ Temperatura Alta',
                        'Ocorrências': len(hot_days),
                        'Última Ocorrência': hot_days['data'].max().strftime('%d/%m/%Y'),
                        'Valor Máximo': f"{hot_days['temp_max'].max():.1f}°C"
                    })
            
            if enable_temp_alerts and 'temp_min' in df.columns:
                cold_days = df[df['temp_min'] < temp_baixa]
                if not cold_days.empty:
                    historical_alerts.append({
                        'Tipo': '🧊 Temperatura Baixa',
                        'Ocorrências': len(cold_days),
                        'Última Ocorrência': cold_days['data'].max().strftime('%d/%m/%Y'),
                        'Valor Mínimo': f"{cold_days['temp_min'].min():.1f}°C"
                    })
            
            if enable_rain_alerts and 'precipitacao_total' in df.columns:
                heavy_rain_days = df[df['precipitacao_total'] > chuva_intensa]
                if not heavy_rain_days.empty:
                    historical_alerts.append({
                        'Tipo': '🌧️ Chuva Intensa',
                        'Ocorrências': len(heavy_rain_days),
                        'Última Ocorrência': heavy_rain_days['data'].max().strftime('%d/%m/%Y'),
                        'Valor Máximo': f"{heavy_rain_days['precipitacao_total'].max():.1f}mm"
                    })
                
                # Verificar períodos secos
                df_sorted = df.sort_values('data')
                current_dry_streak = 0
                max_dry_streak = 0
                
                for _, row in df_sorted.iterrows():
                    if row['precipitacao_total'] == 0:
                        current_dry_streak += 1
                        max_dry_streak = max(max_dry_streak, current_dry_streak)
                    else:
                        current_dry_streak = 0
                
                if max_dry_streak >= dias_secos:
                    historical_alerts.append({
                        'Tipo': '☀️ Período Seco',
                        'Ocorrências': 1,  # Simplificado
                        'Última Ocorrência': 'Histórico',
                        'Valor Máximo': f"{max_dry_streak} dias consecutivos"
                    })
            
            # Exibir alertas históricos
            if historical_alerts:
                st.subheader("📊 Alertas Históricos Encontrados")
                alerts_df = pd.DataFrame(historical_alerts)
                st.dataframe(alerts_df, use_container_width=True, hide_index=True)
                
                # Estatísticas dos alertas
                total_alerts = sum([alert['Ocorrências'] for alert in historical_alerts])
                st.info(f"ℹ️ Total de {total_alerts} eventos de alerta encontrados no histórico")
            else:
                st.success("✅ Nenhum alerta histórico encontrado com os critérios configurados")
        
        # Configurações de notificação
        st.subheader("📧 Configurações de Notificação")
        
        notification_methods = st.multiselect(
            "Métodos de notificação:",
            ["📧 Email", "📱 SMS", "🔔 Push Notification", "📊 Dashboard Alert"],
            default=["📊 Dashboard Alert"]
        )
        
        if notification_methods:
            st.success(f"✅ Alertas serão enviados via: {', '.join(notification_methods)}")
        
        # Horários de alerta
        col1, col2 = st.columns(2)
        
        with col1:
            alert_start_time = st.time_input("Início dos alertas", datetime.strptime("06:00", "%H:%M").time())
        
        with col2:
            alert_end_time = st.time_input("Fim dos alertas", datetime.strptime("22:00", "%H:%M").time())
        
        st.info(f"ℹ️ Alertas serão enviados entre {alert_start_time} e {alert_end_time}")
    
    def _render_business_impact(self, df, store_info):
        """Renderiza análise do impacto climático nos negócios"""
        
        st.subheader("💼 Impacto Climático nos Negócios")
        
        value_col = store_info['value_column']
        
        if value_col not in df.columns:
            st.error(f"❌ Coluna de vendas '{value_col}' não encontrada")
            return
        
        # Análise de correlação clima x vendas
        st.subheader("📊 Correlação Clima x Vendas")
        
        climate_cols = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana']
        available_climate = [col for col in climate_cols if col in df.columns]
        
        correlations = []
        
        for climate_var in available_climate:
            corr = df[value_col].corr(df[climate_var])
            correlations.append({
                'Variável Climática': climate_var.replace('_', ' ').title(),
                'Correlação': corr,
                'Correlação (abs)': abs(corr),
                'Interpretação': self._interpret_correlation(corr)
            })
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('Correlação (abs)', ascending=False)
            
            # Exibir tabela
            display_df = corr_df[['Variável Climática', 'Correlação', 'Interpretação']].copy()
            display_df['Correlação'] = display_df['Correlação'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Gráfico de correlações
            fig_corr = px.bar(
                corr_df,
                x='Variável Climática',
                y='Correlação',
                title="Correlação entre Clima e Vendas",
                color='Correlação',
                color_continuous_scale='RdBu_r'
            )
            fig_corr.update_xaxes(tickangle=45)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Análise por condições climáticas
        st.subheader("🌤️ Vendas por Condições Climáticas")
        
        if 'categoria_temp' in df.columns:
            temp_sales = df.groupby('categoria_temp')[value_col].agg(['mean', 'count']).round(2)
            temp_sales.columns = ['Vendas Médias (R$)', 'Número de Dias']
            
            st.write("**Vendas por Categoria de Temperatura:**")
            st.dataframe(temp_sales, use_container_width=True)
        
        if 'categoria_chuva' in df.columns:
            rain_sales = df.groupby('categoria_chuva')[value_col].agg(['mean', 'count']).round(2)
            rain_sales.columns = ['Vendas Médias (R$)', 'Número de Dias']
            
            st.write("**Vendas por Categoria de Chuva:**")
            st.dataframe(rain_sales, use_container_width=True)
        
        # Insights e recomendações
        self._generate_business_insights(df, value_col, available_climate)
    
    def _interpret_correlation(self, corr):
        """Interpreta valor de correlação"""
        
        abs_corr = abs(corr)
        direction = "Positiva" if corr > 0 else "Negativa"
        
        if abs_corr < 0.1:
            strength = "Muito Fraca"
        elif abs_corr < 0.3:
            strength = "Fraca"
        elif abs_corr < 0.5:
            strength = "Moderada"
        elif abs_corr < 0.7:
            strength = "Forte"
        else:
            strength = "Muito Forte"
        
        return f"{strength} {direction}"
    
    def _generate_business_insights(self, df, value_col, available_climate):
        """Gera insights de negócio baseados no clima"""
        
        st.subheader("💡 Insights e Recomendações")
        
        insights = []
        
        # Análise de temperatura
        if 'temp_media' in available_climate:
            temp_corr = df[value_col].corr(df['temp_media'])
            
            if abs(temp_corr) > 0.3:
                if temp_corr > 0:
                    insights.append({
                        'icon': '🌡️',
                        'title': 'Impacto Positivo da Temperatura',
                        'text': f'Vendas aumentam com temperatura mais alta (correlação: {temp_corr:.3f}). Considere campanhas de verão e produtos sazonais.',
                        'type': 'success'
                    })
                else:
                    insights.append({
                        'icon': '🧊',
                        'title': 'Impacto Negativo da Temperatura',
                        'text': f'Vendas diminuem com temperatura mais alta (correlação: {temp_corr:.3f}). Planeje estratégias para dias quentes.',
                        'type': 'warning'
                    })
        
        # Análise de precipitação
        if 'precipitacao_total' in available_climate:
            rain_corr = df[value_col].corr(df['precipitacao_total'])
            
            if abs(rain_corr) > 0.2:
                if rain_corr < 0:
                    insights.append({
                        'icon': '🌧️',
                        'title': 'Impacto Negativo da Chuva',
                        'text': f'Chuva reduz vendas (correlação: {rain_corr:.3f}). Desenvolva estratégias de delivery e produtos indoor para dias chuvosos.',
                        'type': 'info'
                    })
                else:
                    insights.append({
                        'icon': '☔',
                        'title': 'Impacto Positivo da Chuva',
                        'text': f'Chuva aumenta vendas (correlação: {rain_corr:.3f}). Aproveite dias chuvosos para campanhas especiais.',
                        'type': 'success'
                    })
        
        # Análise sazonal
        if 'mes' in df.columns:
            monthly_sales = df.groupby('mes')[value_col].mean()
            best_month = monthly_sales.idxmax()
            worst_month = monthly_sales.idxmin()
            
            month_names = ['', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                          'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
            
            insights.append({
                'icon': '📅',
                'title': 'Padrão Sazonal',
                'text': f'Melhor mês: {month_names[best_month]} (R$ {monthly_sales[best_month]:,.2f}). Pior mês: {month_names[worst_month]} (R$ {monthly_sales[worst_month]:,.2f}). Ajuste estoque e campanhas sazonalmente.',
                'type': 'info'
            })
        
        # Análise de fim de semana vs clima
        if 'dia_semana' in df.columns and 'precipitacao_total' in available_climate:
            weekend_rain_sales = df[(df['dia_semana'].isin([5, 6])) & (df['precipitacao_total'] > 0)][value_col].mean()
            weekend_norain_sales = df[(df['dia_semana'].isin([5, 6])) & (df['precipitacao_total'] == 0)][value_col].mean()
            
            if not pd.isna(weekend_rain_sales) and not pd.isna(weekend_norain_sales):
                if weekend_norain_sales > weekend_rain_sales * 1.1:
                    insights.append({
                        'icon': '📅',
                        'title': 'Fins de Semana e Clima',
                        'text': 'Fins de semana sem chuva têm vendas significativamente maiores. Monitore previsão do tempo para ajustar estratégias de fim de semana.',
                        'type': 'info'
                    })
        
        # Exibir insights
        if insights:
            for insight in insights:
                if insight['type'] == 'success':
                    st.success(f"{insight['icon']} **{insight['title']}**: {insight['text']}")
                elif insight['type'] == 'warning':
                    st.warning(f"{insight['icon']} **{insight['title']}**: {insight['text']}")
                else:
                    st.info(f"{insight['icon']} **{insight['title']}**: {insight['text']}")
        else:
            st.info("ℹ️ **Correlação Fraca**: O clima não parece ter impacto significativo nas vendas baseado nos dados históricos.")
        
        # Recomendações de ação
        st.subheader("🎯 Plano de Ação")
        
        action_items = [
            "📱 Configure alertas meteorológicos automáticos",
            "📊 Monitore previsão do tempo diariamente",
            "🎯 Ajuste campanhas marketing baseado na previsão",
            "📦 Gerencie estoque considerando sazonalidade climática",
            "🚚 Planeje logística e delivery para dias de mau tempo",
            "👥 Dimensione equipe baseado em padrões climáticos",
            "💰 Ajuste precificação dinâmica baseada no clima",
            "📈 Analise ROI de campanhas por condição climática"
        ]
        
        for item in action_items:
            st.write(f"- {item}")

# Função para integrar com streamlit_app.py
def show_previsao_climatica_page(df, role, store_manager):
    """Função para mostrar a página de previsão climática"""
    
    page = PrevisaoClimaticaPage(store_manager)
    page.render()