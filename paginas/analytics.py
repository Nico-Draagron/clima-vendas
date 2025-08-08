# ============================================================================
# 📈 pages/analytics.py - ANÁLISES AVANÇADAS (INTEGRANDO SEU CÓDIGO EXISTENTE)
# ============================================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from data.store_manager import StoreDataManager
from scipy import stats

class AnalyticsPage:
    """Página de análises avançadas integrando o código existente do usuário"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
    
    def render(self):
        """Renderiza página de análises"""
        
        st.markdown("# 📈 Análises Avançadas")
        st.markdown("**Análises estatísticas profundas e insights de negócio**")
        
        # Seleção de loja
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.warning("⚠️ Nenhuma loja configurada")
            return
        
        # Interface de seleção
        st.subheader("🎯 Configuração da Análise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            store_options = {f"{info['display_name']} ({store_id})": store_id 
                            for store_id, info in stores.items()}
            
            selected_display = st.selectbox(
                "🏪 Escolha a loja:",
                options=list(store_options.keys())
            )
            
            selected_store_id = store_options[selected_display]
        
        with col2:
            analysis_type = st.selectbox(
                "📊 Tipo de Análise:",
                [
                    "🌤️ Clima x Vendas",
                    "📈 Série Temporal", 
                    "🔍 Análise Exploratória",
                    "📊 Correlações Avançadas"
                ]
            )
        
        # Carregar dados da loja selecionada
        df = self.store_manager.load_store_data(selected_store_id)
        
        if df is None or df.empty:
            st.error("❌ Não foi possível carregar dados da loja")
            return
        
        store_info = stores[selected_store_id]
        
        # Renderizar análise selecionada
        if analysis_type == "🌤️ Clima x Vendas":
            self._render_climate_sales_analysis(df, store_info)
        elif analysis_type == "📈 Série Temporal":
            self._render_time_series_analysis(df, store_info)
        elif analysis_type == "🔍 Análise Exploratória":
            self._render_exploratory_analysis(df, store_info)
        elif analysis_type == "📊 Correlações Avançadas":
            self._render_correlation_analysis(df, store_info)
    
    def _render_climate_sales_analysis(self, df, store_info):
        """Análise Clima x Vendas (baseada no seu Clima x Vendas.py)"""
        
        st.subheader("🌤️ Análise Clima x Vendas")
        
        value_col = store_info['value_column']
        
        if value_col not in df.columns:
            st.error(f"❌ Coluna de vendas '{value_col}' não encontrada")
            return
        
        # === FILTROS ===
        st.sidebar.header("🎛️ Filtros")
        
        # Filtro de data (copiado do seu código)
        data_inicio = st.sidebar.date_input("Data inicial", df["data"].min().date())
        data_fim = st.sidebar.date_input("Data final", df["data"].max().date())
        
        # Validação de datas
        if data_inicio > data_fim:
            st.sidebar.error("⚠️ A data inicial não pode ser maior que a data final.")
            return
        
        df_filtered = df[(df["data"] >= pd.to_datetime(data_inicio)) & (df["data"] <= pd.to_datetime(data_fim))]
        
        if df_filtered.empty:
            st.warning("⚠️ Nenhum dado disponível para o intervalo de datas selecionado.")
            return
        
        # === MÉTRICAS PRINCIPAIS ===
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vendas = df_filtered[value_col].fillna(0).sum()
            st.metric("💰 Total de Vendas", f"R$ {total_vendas:,.0f}".replace(",", "."))
        
        with col2:
            media_vendas = df_filtered[value_col].fillna(0).mean()
            st.metric("📊 Média Diária", f"R$ {media_vendas:,.0f}".replace(",", "."))
        
        with col3:
            if 'temp_media' in df_filtered.columns:
                temp_media = df_filtered['temp_media'].fillna(0).mean()
                st.metric("🌡️ Temp. Média", f"{temp_media:.1f}°C")
        
        with col4:
            if 'precipitacao_total' in df_filtered.columns:
                dias_chuva = (df_filtered['precipitacao_total'] > 0).sum()
                st.metric("🌧️ Dias com Chuva", f"{dias_chuva}")
        
        # === ANÁLISES ESPECÍFICAS ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "🌧️ Impacto da Chuva", 
            "🌡️ Temperatura vs Vendas", 
            "📈 Correlações"
        ])
        
        with tab1:
            # Gráfico principal de vendas no tempo
            if 'data' in df_filtered.columns:
                fig = px.line(
                    df_filtered,
                    x='data',
                    y=value_col,
                    title="Evolução das Vendas Diárias"
                )
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas básicas
                st.subheader("📈 Estatísticas do Período")
                stats_df = df_filtered[value_col].describe().to_frame().T
                st.dataframe(stats_df, use_container_width=True)
        
        with tab2:
            # Análise do impacto da chuva (do seu código)
            if 'precipitacao_total' in df_filtered.columns:
                
                # Categorizar chuva
                df_chuva = df_filtered.copy()
                df_chuva['categoria_chuva'] = pd.cut(
                    df_chuva['precipitacao_total'],
                    bins=[-0.1, 0, 10, 50, float('inf')],
                    labels=['Sem chuva', 'Chuva leve', 'Chuva moderada', 'Chuva forte']
                )
                
                # Box plot
                fig = px.box(
                    df_chuva,
                    x='categoria_chuva',
                    y=value_col,
                    title="Distribuição de Vendas por Intensidade de Chuva"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas por categoria
                stats_chuva = df_chuva.groupby('categoria_chuva')[value_col].agg(['mean', 'count', 'std']).round(2)
                stats_chuva.columns = ['Média', 'Contagem', 'Desvio Padrão']
                
                st.subheader("📊 Estatísticas por Categoria de Chuva")
                st.dataframe(stats_chuva, use_container_width=True)
                
                # Teste estatístico
                grupos_chuva = [group[value_col].values for name, group in df_chuva.groupby('categoria_chuva')]
                if len(grupos_chuva) > 1:
                    try:
                        f_stat, p_value = stats.f_oneway(*grupos_chuva)
                        
                        st.subheader("🧪 Teste Estatístico (ANOVA)")
                        
                        if p_value < 0.05:
                            st.success(f"✅ Diferença significativa detectada (p-value: {p_value:.4f})")
                        else:
                            st.info(f"ℹ️ Sem diferença significativa (p-value: {p_value:.4f})")
                    except:
                        st.info("ℹ️ Não foi possível realizar o teste estatístico")
        
        with tab3:
            # Análise de temperatura
            if 'temp_media' in df_filtered.columns:
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot temperatura vs vendas
                    fig = px.scatter(
                        df_filtered,
                        x='temp_media',
                        y=value_col,
                        title="Temperatura Média vs Vendas",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Binning de temperatura
                    df_temp = df_filtered.copy()
                    df_temp['faixa_temp'] = pd.cut(
                        df_temp['temp_media'],
                        bins=5,
                        precision=1
                    )
                    
                    temp_stats = df_temp.groupby('faixa_temp')[value_col].mean().reset_index()
                    
                    fig = px.bar(
                        temp_stats,
                        x='faixa_temp',
                        y=value_col,
                        title="Vendas Médias por Faixa de Temperatura"
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlação temperatura vs vendas
                corr_temp = df_filtered['temp_media'].corr(df_filtered[value_col])
                st.metric("🔗 Correlação Temperatura-Vendas", f"{corr_temp:.3f}")
        
        with tab4:
            # Matriz de correlação avançada
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df_filtered[numeric_cols].corr()
                
                # Heatmap de correlação
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlação - Todas as Variáveis",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlações com vendas
                if value_col in corr_matrix.columns:
                    vendas_corr = corr_matrix[value_col].drop(value_col).sort_values(key=abs, ascending=False)
                    
                    st.subheader("🎯 Top Correlações com Vendas")
                    
                    for var, corr in vendas_corr.head(5).items():
                        emoji = "📈" if corr > 0 else "📉"
                        strength = "forte" if abs(corr) > 0.5 else "moderada" if abs(corr) > 0.3 else "fraca"
                        st.write(f"{emoji} **{var}**: {corr:.3f} (correlação {strength})")
    
    def _render_time_series_analysis(self, df, store_info):
        """Análise de Série Temporal (baseada no seu Serie_Temporal.py)"""
        
        st.subheader("📈 Análise de Série Temporal")
        
        value_col = store_info['value_column']
        
        # === CONFIGURAÇÕES ===
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Seleção de variável (do seu código)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            var_selecionada = st.selectbox(
                "📊 Escolha a variável:",
                numeric_cols,
                index=numeric_cols.index(value_col) if value_col in numeric_cols else 0
            )
        
        with col2:
            # Janela da média móvel
            janela_movel = st.slider(
                "📊 Janela da média móvel (dias):",
                min_value=3,
                max_value=30,
                value=7
            )
        
        with col3:
            # Mostrar dados originais
            mostrar_original = st.checkbox("👁️ Mostrar série original", value=True)
        
        if var_selecionada not in df.columns:
            st.error(f"❌ Variável '{var_selecionada}' não encontrada")
            return
        
        # === CÁLCULOS ===
        df_ts = df.copy()
        df_ts = df_ts.sort_values('data')
        
        # Calcular média móvel (do seu código)
        df_ts[f'{var_selecionada}_movel'] = df_ts[var_selecionada].rolling(window=janela_movel).mean()
        
        # === VISUALIZAÇÃO PRINCIPAL ===
        fig = go.Figure()
        
        # Série original
        if mostrar_original:
            fig.add_trace(go.Scatter(
                x=df_ts['data'],
                y=df_ts[var_selecionada],
                mode='lines',
                name='Original',
                line=dict(width=1, color='lightblue'),
                opacity=0.6
            ))
        
        # Média móvel
        fig.add_trace(go.Scatter(
            x=df_ts['data'],
            y=df_ts[f'{var_selecionada}_movel'],
            mode='lines',
            name=f'Média Móvel ({janela_movel} dias)',
            line=dict(width=3, color='darkblue')
        ))
        
        fig.update_layout(
            title=f"Análise Temporal: {var_selecionada}",
            xaxis_title="Data",
            yaxis_title=var_selecionada,
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # === ANÁLISES ESTATÍSTICAS ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Estatísticas Descritivas")
            stats = df[var_selecionada].describe()
            
            for stat, value in stats.items():
                st.write(f"**{stat.title()}:** {value:.2f}")
        
        with col2:
            st.subheader("📈 Análise de Tendência")
            
            # Calcular tendência (do seu código adaptado)
            if len(df_ts) > 1:
                try:
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        range(len(df_ts)), df_ts[var_selecionada].fillna(0)
                    )
                    
                    if slope > 0:
                        st.success(f"📈 Tendência crescente (slope: {slope:.4f})")
                    elif slope < 0:
                        st.error(f"📉 Tendência decrescente (slope: {slope:.4f})")
                    else:
                        st.info("➡️ Tendência estável")
                    
                    st.write(f"**R²:** {r_value**2:.3f}")
                    st.write(f"**P-value:** {p_value:.4f}")
                    
                except ImportError:
                    st.info("Scipy não disponível para análise de tendência")
        
        # === DECOMPOSIÇÃO SAZONAL ===
        st.subheader("🔄 Análise Sazonal")
        
        if len(df_ts) > 30:  # Necessário período mínimo
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Análise por dia da semana
                df_ts['dia_semana'] = df_ts['data'].dt.day_name()
                weekday_stats = df_ts.groupby('dia_semana')[var_selecionada].mean().reset_index()
                
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_stats['dia_semana'] = pd.Categorical(
                    weekday_stats['dia_semana'], 
                    categories=weekday_order, 
                    ordered=True
                )
                weekday_stats = weekday_stats.sort_values('dia_semana')
                
                fig_week = px.bar(
                    weekday_stats,
                    x='dia_semana',
                    y=var_selecionada,
                    title="Padrão Semanal"
                )
                fig_week.update_xaxis(tickangle=45)
                st.plotly_chart(fig_week, use_container_width=True)
            
            with col2:
                # Análise por mês
                df_ts['mes'] = df_ts['data'].dt.month
                monthly_stats = df_ts.groupby('mes')[var_selecionada].mean().reset_index()
                
                month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                              7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
                monthly_stats['mes_nome'] = monthly_stats['mes'].map(month_names)
                
                fig_month = px.bar(
                    monthly_stats,
                    x='mes_nome',
                    y=var_selecionada,
                    title="Padrão Mensal"
                )
                st.plotly_chart(fig_month, use_container_width=True)
        
        # === DADOS RECENTES ===
        st.subheader("📋 Últimos Registros")
        
        # Mostrar últimos 10 registros (do seu código)
        colunas_exibir = ['data', var_selecionada]
        if f'{var_selecionada}_movel' in df_ts.columns:
            colunas_exibir.append(f'{var_selecionada}_movel')
        
        df_recente = df_ts[colunas_exibir].tail(10).copy()
        df_recente['data'] = df_recente['data'].dt.strftime('%d/%m/%Y')
        
        st.dataframe(df_recente, use_container_width=True, hide_index=True)
        
        # === DOWNLOAD ===
        st.subheader("💾 Download dos Dados")
        
        # Preparar dados para download (do seu código)
        df_download = df_ts.copy()
        df_download[f'{var_selecionada}_media_movel_{janela_movel}d'] = df_download[f'{var_selecionada}_movel']
        
        csv_download = df_download.to_csv(index=False)
        
        st.download_button(
            label="📥 Baixar dados processados (CSV)",
            data=csv_download,
            file_name=f"dados_processados_{var_selecionada}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _render_exploratory_analysis(self, df, store_info):
        """Análise exploratória avançada"""
        
        st.subheader("🔍 Análise Exploratória")
        
        # Informações gerais do dataset
        st.subheader("📊 Informações Gerais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Registros", len(df))
        
        with col2:
            st.metric("🔢 Colunas", len(df.columns))
        
        with col3:
            missing_count = df.isnull().sum().sum()
            st.metric("❓ Valores Faltantes", missing_count)
        
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Colunas Numéricas", numeric_cols)
        
        # Qualidade dos dados
        st.subheader("🔍 Qualidade dos Dados")
        
        # Dados faltantes por coluna
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        quality_df = pd.DataFrame({
            'Coluna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Percentual (%)': missing_pct.values
        })
        quality_df = quality_df[quality_df['Valores Faltantes'] > 0].sort_values('Percentual (%)', ascending=False)
        
        if not quality_df.empty:
            st.dataframe(quality_df, use_container_width=True)
        else:
            st.success("✅ Nenhum valor faltante encontrado!")
        
        # Distribuições das variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.subheader("📈 Distribuições das Variáveis")
            
            selected_vars = st.multiselect(
                "Escolha variáveis para análise:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_vars:
                for var in selected_vars:
                    with st.expander(f"📊 Distribuição: {var}"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histograma
                            fig_hist = px.histogram(
                                df,
                                x=var,
                                nbins=30,
                                title=f"Histograma - {var}"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig_box = px.box(
                                df,
                                y=var,
                                title=f"Box Plot - {var}"
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Estatísticas
                        st.write("**Estatísticas:**")
                        stats = df[var].describe()
                        stats_df = pd.DataFrame(stats).T
                        st.dataframe(stats_df, use_container_width=True)
    
    def _render_correlation_analysis(self, df, store_info):
        """Análise de correlações avançada"""
        
        st.subheader("📊 Análise de Correlações Avançada")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Poucas variáveis numéricas para análise de correlação")
            return
        
        # Seleção de variáveis
        selected_vars = st.multiselect(
            "Escolha variáveis para análise:",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        )
        
        if len(selected_vars) < 2:
            st.warning("⚠️ Selecione pelo menos 2 variáveis")
            return
        
        # Calcular matriz de correlação
        corr_matrix = df[selected_vars].corr()
        
        # Heatmap interativo
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de correlações mais fortes
        st.subheader("🎯 Correlações Mais Fortes")
        
        # Extrair correlações (excluindo diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_value):
                    corr_pairs.append({
                        'Variável 1': var1,
                        'Variável 2': var2,
                        'Correlação': corr_value,
                        'Correlação Abs': abs(corr_value)
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlação Abs', ascending=False)
            
            # Top correlações
            st.dataframe(corr_df.head(10), use_container_width=True)
            
            # Scatter plots das correlações mais fortes
            st.subheader("📈 Scatter Plots - Top Correlações")
            
            for i, row in corr_df.head(3).iterrows():
                var1, var2 = row['Variável 1'], row['Variável 2']
                corr_val = row['Correlação']
                
                fig_scatter = px.scatter(
                    df,
                    x=var1,
                    y=var2,
                    title=f"{var1} vs {var2} (Correlação: {corr_val:.3f})",
                    trendline="ols"
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
