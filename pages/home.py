# ============================================================================
# 🏠 pages/home.py - DASHBOARD EXECUTIVO
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.store_manager import StoreDataManager
import numpy as np

class HomePage:
    """Dashboard executivo com métricas das lojas"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
    
    def render(self):
        """Renderiza dashboard principal"""
        
        st.markdown("# 🏠 Dashboard Executivo")
        st.markdown("**Visão geral do desempenho de todas as lojas**")
        
        # Carregar dados das lojas
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.warning("⚠️ Nenhuma loja configurada no sistema")
            return
        
        # Carregar dados de todas as lojas
        store_data = {}
        total_records = 0
        
        for store_id, store_info in stores.items():
            df = self.store_manager.load_store_data(store_id)
            if df is not None and not df.empty:
                store_data[store_id] = {
                    'data': df,
                    'info': store_info,
                    'records': len(df)
                }
                total_records += len(df)
        
        if not store_data:
            st.error("❌ Não foi possível carregar dados de nenhuma loja")
            return
        
        # === MÉTRICAS PRINCIPAIS ===
        self._render_main_metrics(store_data, total_records)
        
        # === GRÁFICOS COMPARATIVOS ===
        self._render_comparative_charts(store_data)
        
        # === ANÁLISE TEMPORAL ===
        self._render_temporal_analysis(store_data)
        
        # === RESUMO POR LOJA ===
        self._render_store_summary(store_data)
    
    def _render_main_metrics(self, store_data, total_records):
        """Renderiza métricas principais"""
        
        st.subheader("📊 Métricas Gerais")
        
        # Calcular métricas agregadas
        total_sales = 0
        total_days = 0
        avg_temp = 0
        rainy_days = 0
        
        for store_id, data in store_data.items():
            df = data['data']
            value_col = data['info']['value_column']
            
            if value_col in df.columns:
                store_sales = df[value_col].fillna(0).sum()
                total_sales += store_sales
            
            if 'temp_media' in df.columns:
                avg_temp += df['temp_media'].fillna(0).mean()
            
            if 'precipitacao_total' in df.columns:
                rainy_days += (df['precipitacao_total'] > 0).sum()
        
        avg_temp = avg_temp / len(store_data) if store_data else 0
        
        # Exibir métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Vendas Totais",
                f"R$ {total_sales:,.0f}".replace(",", "."),
                help="Soma de todas as vendas de todas as lojas"
            )
        
        with col2:
            st.metric(
                "🏪 Lojas Ativas", 
                len(store_data),
                help="Número de lojas com dados disponíveis"
            )
        
        with col3:
            st.metric(
                "🌡️ Temp. Média",
                f"{avg_temp:.1f}°C",
                help="Temperatura média de todas as lojas"
            )
        
        with col4:
            st.metric(
                "🌧️ Dias de Chuva",
                f"{rainy_days}",
                help="Total de dias com precipitação > 0"
            )
    
    def _render_comparative_charts(self, store_data):
        """Renderiza gráficos comparativos entre lojas"""
        
        st.subheader("📈 Comparativo Entre Lojas")
        
        # Preparar dados para comparação
        comparison_data = []
        
        for store_id, data in store_data.items():
            df = data['data']
            store_name = data['info']['display_name']
            value_col = data['info']['value_column']
            
            if value_col in df.columns and 'data' in df.columns:
                # Agrupar por mês
                df_monthly = df.copy()
                df_monthly['ano_mes'] = df_monthly['data'].dt.to_period('M')
                monthly_sales = df_monthly.groupby('ano_mes')[value_col].sum().reset_index()
                monthly_sales['loja'] = store_name
                monthly_sales['ano_mes_str'] = monthly_sales['ano_mes'].astype(str)
                
                comparison_data.append(monthly_sales)
        
        if comparison_data:
            # Combinar dados de todas as lojas
            df_combined = pd.concat(comparison_data, ignore_index=True)
            
            # Gráfico de linha comparativo
            fig = px.line(
                df_combined,
                x='ano_mes_str',
                y=value_col,
                color='loja',
                title="Evolução Mensal das Vendas por Loja",
                labels={
                    'ano_mes_str': 'Período',
                    value_col: 'Vendas (R$)',
                    'loja': 'Loja'
                }
            )
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de barras - Total por loja
            col1, col2 = st.columns(2)
            
            with col1:
                total_by_store = df_combined.groupby('loja')[value_col].sum().reset_index()
                
                fig_bar = px.bar(
                    total_by_store,
                    x='loja',
                    y=value_col,
                    title="Total de Vendas por Loja",
                    labels={'loja': 'Loja', value_col: 'Vendas (R$)'}
                )
                
                fig_bar.update_layout(height=350)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                avg_by_store = df_combined.groupby('loja')[value_col].mean().reset_index()
                
                fig_avg = px.bar(
                    avg_by_store,
                    x='loja', 
                    y=value_col,
                    title="Média Mensal por Loja",
                    labels={'loja': 'Loja', value_col: 'Média (R$)'},
                    color=value_col,
                    color_continuous_scale='viridis'
                )
                
                fig_avg.update_layout(height=350)
                st.plotly_chart(fig_avg, use_container_width=True)
    
    def _render_temporal_analysis(self, store_data):
        """Renderiza análise temporal agregada"""
        
        st.subheader("⏰ Análise Temporal Agregada")
        
        # Combinar dados de todas as lojas por data
        all_data = []
        
        for store_id, data in store_data.items():
            df = data['data'].copy()
            value_col = data['info']['value_column']
            
            if value_col in df.columns:
                df_temp = df[['data', value_col]].copy()
                df_temp['loja'] = data['info']['display_name']
                all_data.append(df_temp)
        
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            
            # Agregação diária (soma de todas as lojas)
            daily_total = df_all.groupby('data')[value_col].sum().reset_index()
            
            # Análise de sazonalidade
            daily_total['dia_semana'] = daily_total['data'].dt.day_name()
            daily_total['mes'] = daily_total['data'].dt.month
            daily_total['ano'] = daily_total['data'].dt.year
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vendas por dia da semana
                weekday_avg = daily_total.groupby('dia_semana')[value_col].mean().reset_index()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_avg['dia_semana'] = pd.Categorical(weekday_avg['dia_semana'], categories=weekday_order, ordered=True)
                weekday_avg = weekday_avg.sort_values('dia_semana')
                
                fig_weekday = px.bar(
                    weekday_avg,
                    x='dia_semana',
                    y=value_col,
                    title="Média de Vendas por Dia da Semana",
                    labels={'dia_semana': 'Dia da Semana', value_col: 'Vendas Médias (R$)'}
                )
                
                fig_weekday.update_layout(height=300)
                st.plotly_chart(fig_weekday, use_container_width=True)
            
            with col2:
                # Vendas por mês
                monthly_avg = daily_total.groupby('mes')[value_col].mean().reset_index()
                month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                              7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
                monthly_avg['mes_nome'] = monthly_avg['mes'].map(month_names)
                
                fig_month = px.bar(
                    monthly_avg,
                    x='mes_nome',
                    y=value_col,
                    title="Média de Vendas por Mês",
                    labels={'mes_nome': 'Mês', value_col: 'Vendas Médias (R$)'}
                )
                
                fig_month.update_layout(height=300)
                st.plotly_chart(fig_month, use_container_width=True)
    
    def _render_store_summary(self, store_data):
        """Renderiza resumo detalhado por loja"""
        
        st.subheader("🏪 Resumo Detalhado por Loja")
        
        for store_id, data in store_data.items():
            df = data['data']
            store_info = data['info']
            value_col = store_info['value_column']
            
            with st.expander(f"📊 {store_info['display_name']} ({store_id})"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 Métricas de Vendas**")
                    if value_col in df.columns:
                        total = df[value_col].fillna(0).sum()
                        media = df[value_col].fillna(0).mean()
                        maximo = df[value_col].fillna(0).max()
                        
                        st.write(f"**Total:** R$ {total:,.0f}".replace(",", "."))
                        st.write(f"**Média:** R$ {media:,.0f}".replace(",", "."))
                        st.write(f"**Máximo:** R$ {maximo:,.0f}".replace(",", "."))
                
                with col2:
                    st.markdown("**🌡️ Condições Climáticas**")
                    if 'temp_media' in df.columns:
                        temp_avg = df['temp_media'].fillna(0).mean()
                        temp_max = df['temp_max'].fillna(0).max() if 'temp_max' in df.columns else 0
                        temp_min = df['temp_min'].fillna(0).min() if 'temp_min' in df.columns else 0
                        
                        st.write(f"**Temp. Média:** {temp_avg:.1f}°C")
                        st.write(f"**Temp. Máxima:** {temp_max:.1f}°C")
                        st.write(f"**Temp. Mínima:** {temp_min:.1f}°C")
                
                with col3:
                    st.markdown("**📊 Informações Gerais**")
                    period_start = df['data'].min().strftime('%d/%m/%Y') if 'data' in df.columns else 'N/A'
                    period_end = df['data'].max().strftime('%d/%m/%Y') if 'data' in df.columns else 'N/A'
                    
                    st.write(f"**Registros:** {len(df)}")
                    st.write(f"**Período:** {period_start}")
                    st.write(f"**até {period_end}")