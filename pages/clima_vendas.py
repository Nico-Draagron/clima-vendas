# ============================================================================
# 🌤️ pages/clima_vendas.py - ANÁLISE CLIMA x VENDAS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ClimaVendasPage:
    """Página completa de análise clima x vendas"""
    
    def __init__(self, store_manager):
        self.store_manager = store_manager
    
    def render(self):
        """Renderiza página principal"""
        
        st.markdown("# 🌤️ Análise Clima x Vendas")
        st.markdown("**Descubra como o clima impacta suas vendas e tome decisões baseadas em dados**")
        
        # Carregar dados
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.error("❌ Nenhuma loja configurada. Configure uma loja no painel administrativo.")
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
            st.error(f"❌ Coluna de vendas '{value_col}' não encontrada nos dados")
            return
        
        # Preparar dados
        df = self._prepare_data(df, value_col)
        
        if df is None:
            return
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visão Geral",
            "🔍 Correlações", 
            "📈 Análise Detalhada",
            "🌡️ Impacto da Temperatura",
            "🌧️ Impacto da Precipitação"
        ])
        
        with tab1:
            self._render_overview(df, value_col, store_info['display_name'])
        
        with tab2:
            self._render_correlations(df, value_col)
        
        with tab3:
            self._render_detailed_analysis(df, value_col)
        
        with tab4:
            self._render_temperature_analysis(df, value_col)
        
        with tab5:
            self._render_precipitation_analysis(df, value_col)
    
    def _prepare_data(self, df, value_col):
        """Prepara e valida os dados"""
        
        # Verificar colunas essenciais
        required_cols = ['data', value_col]
        climate_cols = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Colunas obrigatórias faltando: {missing_cols}")
            return None
        
        # Converter data
        df['data'] = pd.to_datetime(df['data'])
        
        # Verificar dados climáticos disponíveis
        available_climate = [col for col in climate_cols if col in df.columns]
        
        if not available_climate:
            st.error("❌ Nenhuma variável climática encontrada nos dados")
            return None
        
        # Filtros de data
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "📅 Data Inicial",
                value=df['data'].min().date(),
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
        
        # Aplicar filtros
        df_filtered = df[
            (df['data'] >= pd.to_datetime(start_date)) & 
            (df['data'] <= pd.to_datetime(end_date))
        ].copy()
        
        if df_filtered.empty:
            st.warning("⚠️ Nenhum dado encontrado para o período selecionado")
            return None
        
        # Adicionar features temporais
        df_filtered = self._add_temporal_features(df_filtered)
        
        return df_filtered
    
    def _add_temporal_features(self, df):
        """Adiciona features temporais aos dados"""
        
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_mes'] = df['data'].dt.day
        df['dia_ano'] = df['data'].dt.dayofyear
        df['semana_ano'] = df['data'].dt.isocalendar().week
        
        # Categorias temporais
        df['trimestre'] = df['data'].dt.quarter
        df['eh_weekend'] = df['dia_semana'].isin([5, 6])
        df['estacao'] = df['mes'].map({
            12: 'Verão', 1: 'Verão', 2: 'Verão',
            3: 'Outono', 4: 'Outono', 5: 'Outono', 
            6: 'Inverno', 7: 'Inverno', 8: 'Inverno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        })
        
        return df
    
    def _render_overview(self, df, value_col, store_name):
        """Visão geral com métricas principais"""
        
        st.subheader(f"📊 Visão Geral - {store_name}")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vendas = df[value_col].sum()
            st.metric("💰 Vendas Totais", f"R$ {total_vendas:,.2f}".replace(',', '.'))
        
        with col2:
            media_vendas = df[value_col].mean()
            st.metric("📊 Média Diária", f"R$ {media_vendas:,.2f}".replace(',', '.'))
        
        with col3:
            if 'precipitacao_total' in df.columns:
                dias_chuva = (df['precipitacao_total'] > 0).sum()
                pct_chuva = (dias_chuva / len(df)) * 100
                st.metric("🌧️ Dias com Chuva", f"{dias_chuva} ({pct_chuva:.1f}%)")
            else:
                st.metric("🌧️ Dados de Chuva", "N/A")
        
        with col4:
            if 'temp_media' in df.columns:
                temp_media = df['temp_media'].mean()
                st.metric("🌡️ Temp. Média", f"{temp_media:.1f}°C")
            else:
                st.metric("🌡️ Temperatura", "N/A")
        
        # Gráfico de vendas no tempo com overlay climático
        st.subheader("📈 Evolução das Vendas vs Clima")
        
        if 'temp_media' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Vendas Diárias', 'Temperatura e Precipitação'],
                specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                vertical_spacing=0.1
            )
            
            # Vendas
            fig.add_trace(
                go.Scatter(
                    x=df['data'], 
                    y=df[value_col],
                    name='Vendas',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Temperatura
            fig.add_trace(
                go.Scatter(
                    x=df['data'], 
                    y=df['temp_media'],
                    name='Temperatura',
                    line=dict(color='#ff7f0e', width=1),
                    hovertemplate='<b>%{x}</b><br>Temp: %{y:.1f}°C<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Precipitação (se disponível)
            if 'precipitacao_total' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df['data'], 
                        y=df['precipitacao_total'],
                        name='Precipitação',
                        marker_color='#2ca02c',
                        opacity=0.6,
                        hovertemplate='<b>%{x}</b><br>Chuva: %{y:.1f}mm<extra></extra>'
                    ),
                    row=2, col=1, secondary_y=True
                )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Análise Temporal: Vendas e Variáveis Climáticas"
            )
            
            fig.update_xaxes(title_text="Data", row=2, col=1)
            fig.update_yaxes(title_text="Vendas (R$)", row=1, col=1)
            fig.update_yaxes(title_text="Temperatura (°C)", row=2, col=1)
            
            if 'precipitacao_total' in df.columns:
                fig.update_yaxes(title_text="Precipitação (mm)", row=2, col=1, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights rápidos
        self._render_quick_insights(df, value_col)
    
    def _render_quick_insights(self, df, value_col):
        """Insights rápidos automatizados"""
        
        st.subheader("💡 Insights Automáticos")
        
        insights = []
        
        # Insight 1: Melhor dia da semana
        vendas_dia_semana = df.groupby('dia_semana')[value_col].mean()
        melhor_dia = vendas_dia_semana.idxmax()
        dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        
        insights.append({
            'icon': '📅',
            'title': 'Melhor Dia da Semana',
            'text': f'{dias_semana[melhor_dia]} tem as maiores vendas médias (R$ {vendas_dia_semana[melhor_dia]:,.2f})'
        })
        
        # Insight 2: Impacto da temperatura
        if 'temp_media' in df.columns:
            corr_temp = df[value_col].corr(df['temp_media'])
            if abs(corr_temp) > 0.3:
                relacao = "positiva" if corr_temp > 0 else "negativa"
                insights.append({
                    'icon': '🌡️',
                    'title': 'Impacto da Temperatura',
                    'text': f'Correlação {relacao} forte entre temperatura e vendas (r={corr_temp:.3f})'
                })
        
        # Insight 3: Impacto da chuva
        if 'precipitacao_total' in df.columns:
            vendas_com_chuva = df[df['precipitacao_total'] > 0][value_col].mean()
            vendas_sem_chuva = df[df['precipitacao_total'] == 0][value_col].mean()
            
            if vendas_sem_chuva > 0:
                impacto_pct = ((vendas_com_chuva - vendas_sem_chuva) / vendas_sem_chuva) * 100
                
                if abs(impacto_pct) > 5:
                    sinal = "aumentam" if impacto_pct > 0 else "diminuem"
                    insights.append({
                        'icon': '🌧️',
                        'title': 'Impacto da Chuva',
                        'text': f'Vendas {sinal} {abs(impacto_pct):.1f}% em dias chuvosos'
                    })
        
        # Insight 4: Sazonalidade
        vendas_estacao = df.groupby('estacao')[value_col].mean()
        melhor_estacao = vendas_estacao.idxmax()
        pior_estacao = vendas_estacao.idxmin()
        
        insights.append({
            'icon': '🍂',
            'title': 'Sazonalidade',
            'text': f'Melhor época: {melhor_estacao} (R$ {vendas_estacao[melhor_estacao]:,.2f}). Pior: {pior_estacao} (R$ {vendas_estacao[pior_estacao]:,.2f})'
        })
        
        # Exibir insights
        for insight in insights:
            st.info(f"{insight['icon']} **{insight['title']}**: {insight['text']}")
    
    def _render_correlations(self, df, value_col):
        """Análise de correlações"""
        
        st.subheader("🔍 Matrix de Correlações")
        
        # Selecionar colunas numéricas relevantes
        numeric_cols = [value_col]
        climate_cols = ['temp_media', 'temp_max', 'temp_min', 'precipitacao_total', 'umid_mediana', 'rad_mediana']
        temporal_cols = ['mes', 'dia_semana', 'dia_ano']
        
        available_cols = [col for col in climate_cols + temporal_cols if col in df.columns]
        correlation_cols = numeric_cols + available_cols
        
        if len(correlation_cols) < 2:
            st.warning("⚠️ Dados insuficientes para análise de correlação")
            return
        
        # Calcular correlações
        corr_matrix = df[correlation_cols].corr()
        
        # Heatmap de correlações
        fig_heatmap = px.imshow(
            corr_matrix,
            labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Matrix de Correlação: Vendas vs Variáveis Climáticas"
        )
        
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Correlações mais importantes
        st.subheader("🎯 Correlações Mais Relevantes")
        
        vendas_corr = corr_matrix[value_col].drop(value_col).abs().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 Correlações Mais Fortes:**")
            for var, corr_val in vendas_corr.head(5).items():
                original_corr = corr_matrix[value_col][var]
                sinal = "📈" if original_corr > 0 else "📉"
                st.write(f"{sinal} **{var}**: {original_corr:.3f}")
        
        with col2:
            st.write("**🧊 Correlações Mais Fracas:**")
            for var, corr_val in vendas_corr.tail(5).items():
                original_corr = corr_matrix[value_col][var]
                st.write(f"• **{var}**: {original_corr:.3f}")
        
        # Testes estatísticos
        if len(available_cols) > 0:
            st.subheader("📊 Testes de Significância")
            
            for col in available_cols[:3]:  # Limitar a 3 para não sobrecarregar
                if col in df.columns:
                    corr_pearson, p_value = pearsonr(df[value_col], df[col])
                    
                    significancia = "✅ Significativa" if p_value < 0.05 else "❌ Não significativa"
                    
                    st.write(f"**{col} vs Vendas:**")
                    st.write(f"• Correlação: {corr_pearson:.3f}")
                    st.write(f"• P-valor: {p_value:.4f}")
                    st.write(f"• Significância: {significancia}")
                    st.markdown("---")
    
    def _render_detailed_analysis(self, df, value_col):
        """Análise detalhada com segmentações"""
        
        st.subheader("📈 Análise Detalhada por Segmentos")
        
        # Análise por estação do ano
        if 'estacao' in df.columns:
            st.subheader("🍂 Vendas por Estação")
            
            vendas_estacao = df.groupby('estacao')[value_col].agg(['mean', 'std', 'count']).round(2)
            vendas_estacao.columns = ['Média', 'Desvio Padrão', 'Dias']
            
            st.dataframe(vendas_estacao, use_container_width=True)
            
            # Boxplot por estação
            fig_box = px.box(
                df, 
                x='estacao', 
                y=value_col,
                title="Distribuição de Vendas por Estação",
                labels={'estacao': 'Estação', value_col: 'Vendas (R$)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Análise por dia da semana
        st.subheader("📅 Vendas por Dia da Semana")
        
        dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        df_dia_semana = df.copy()
        df_dia_semana['nome_dia'] = df_dia_semana['dia_semana'].map(dict(enumerate(dias_semana)))
        
        vendas_dia = df_dia_semana.groupby('nome_dia')[value_col].agg(['mean', 'std']).round(2)
        vendas_dia = vendas_dia.reindex(dias_semana)
        
        fig_dia = px.bar(
            x=vendas_dia.index,
            y=vendas_dia['mean'],
            error_y=vendas_dia['std'],
            title="Vendas Médias por Dia da Semana",
            labels={'x': 'Dia da Semana', 'y': 'Vendas Médias (R$)'}
        )
        st.plotly_chart(fig_dia, use_container_width=True)
        
        # Clustering de padrões climáticos
        if all(col in df.columns for col in ['temp_media', 'precipitacao_total']):
            st.subheader("🎯 Clusters Climáticos")
            
            # Preparar dados para clustering
            features_cluster = df[['temp_media', 'precipitacao_total']].fillna(df[['temp_media', 'precipitacao_total']].mean())
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_cluster)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            df_cluster = df.copy()
            df_cluster['cluster'] = clusters
            df_cluster['cluster_nome'] = df_cluster['cluster'].map({
                0: 'Clima Moderado',
                1: 'Clima Quente/Seco', 
                2: 'Clima Frio/Úmido'
            })
            
            # Vendas por cluster
            vendas_cluster = df_cluster.groupby('cluster_nome')[value_col].agg(['mean', 'count']).round(2)
            vendas_cluster.columns = ['Vendas Médias', 'Número de Dias']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Vendas por Padrão Climático:**")
                st.dataframe(vendas_cluster, use_container_width=True)
            
            with col2:
                # Scatter plot dos clusters
                fig_cluster = px.scatter(
                    df_cluster,
                    x='temp_media',
                    y='precipitacao_total',
                    color='cluster_nome',
                    size=value_col,
                    title="Padrões Climáticos e Vendas",
                    labels={
                        'temp_media': 'Temperatura Média (°C)',
                        'precipitacao_total': 'Precipitação (mm)',
                        value_col: 'Vendas (R$)'
                    }
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
    
    def _render_temperature_analysis(self, df, value_col):
        """Análise específica do impacto da temperatura"""
        
        st.subheader("🌡️ Impacto Detalhado da Temperatura")
        
        if 'temp_media' not in df.columns:
            st.warning("⚠️ Dados de temperatura não disponíveis")
            return
        
        # Categorização por faixas de temperatura
        df_temp = df.copy()
        df_temp['faixa_temp'] = pd.cut(
            df_temp['temp_media'],
            bins=[0, 18, 22, 26, 30, 50],
            labels=['Muito Frio (<18°C)', 'Frio (18-22°C)', 'Ameno (22-26°C)', 'Quente (26-30°C)', 'Muito Quente (>30°C)']
        )
        
        # Estatísticas por faixa
        temp_stats = df_temp.groupby('faixa_temp')[value_col].agg(['mean', 'std', 'count']).round(2)
        temp_stats.columns = ['Vendas Médias', 'Desvio Padrão', 'Número de Dias']
        
        st.write("**📊 Vendas por Faixa de Temperatura:**")
        st.dataframe(temp_stats, use_container_width=True)
        
        # Gráficos detalhados
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot temperatura vs vendas
            fig_scatter = px.scatter(
                df_temp,
                x='temp_media',
                y=value_col,
                trendline="ols",
                title="Relação Temperatura vs Vendas",
                labels={'temp_media': 'Temperatura Média (°C)', value_col: 'Vendas (R$)'}
            )
            
            # Adicionar linha de tendência manual se necessário
            z = np.polyfit(df_temp['temp_media'], df_temp[value_col], 1)
            trendline_text = f"Tendência: y = {z[0]:.2f}x + {z[1]:.2f}"
            
            fig_scatter.add_annotation(
                x=df_temp['temp_media'].max() * 0.7,
                y=df_temp[value_col].max() * 0.9,
                text=trendline_text,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)"
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Violin plot por faixa de temperatura
            fig_violin = px.violin(
                df_temp.dropna(subset=['faixa_temp']),
                x='faixa_temp',
                y=value_col,
                title="Distribuição de Vendas por Faixa de Temperatura",
                labels={'faixa_temp': 'Faixa de Temperatura', value_col: 'Vendas (R$)'}
            )
            fig_violin.update_xaxes(tickangle=45)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # Análise de temperatura ótima
        st.subheader("🎯 Temperatura Ótima para Vendas")
        
        # Encontrar faixa ótima
        melhor_faixa = temp_stats['Vendas Médias'].idxmax()
        vendas_otima = temp_stats.loc[melhor_faixa, 'Vendas Médias']
        
        # Correlação temperatura vs vendas
        corr_temp = df['temp_media'].corr(df[value_col])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌡️ Faixa Ótima", melhor_faixa)
        
        with col2:
            st.metric("💰 Vendas Médias", f"R$ {vendas_otima:,.2f}".replace(',', '.'))
        
        with col3:
            st.metric("📊 Correlação", f"{corr_temp:.3f}")
        
        # Recomendações baseadas na temperatura
        self._render_temperature_recommendations(df_temp, value_col, corr_temp)
    
    def _render_temperature_recommendations(self, df, value_col, correlation):
        """Recomendações baseadas na análise de temperatura"""
        
        st.subheader("💡 Recomendações - Temperatura")
        
        recommendations = []
        
        if abs(correlation) > 0.3:
            if correlation > 0:
                recommendations.append({
                    'type': 'success',
                    'text': f"📈 **Correlação Positiva Forte** (r={correlation:.3f}): Vendas aumentam com temperaturas mais altas. Considere campanhas promocionais em dias quentes."
                })
            else:
                recommendations.append({
                    'type': 'info', 
                    'text': f"📉 **Correlação Negativa Forte** (r={correlation:.3f}): Vendas diminuem com temperaturas mais altas. Planeje estratégias para dias quentes."
                })
        
        # Análise de variabilidade
        temp_variance = df['temp_media'].var()
        if temp_variance > 25:  # Alta variabilidade de temperatura
            recommendations.append({
                'type': 'warning',
                'text': "🌡️ **Alta Variabilidade Térmica**: Grande variação de temperatura detectada. Mantenha estratégias flexíveis para diferentes condições climáticas."
            })
        
        # Análise sazonal
        if 'estacao' in df.columns:
            vendas_verao = df[df['estacao'] == 'Verão'][value_col].mean()
            vendas_inverno = df[df['estacao'] == 'Inverno'][value_col].mean()
            
            if vendas_verao > vendas_inverno * 1.1:
                recommendations.append({
                    'type': 'success',
                    'text': f"☀️ **Padrão Sazonal**: Vendas 10%+ maiores no verão (R$ {vendas_verao:,.2f} vs R$ {vendas_inverno:,.2f}). Intensifique campanhas de verão."
                })
        
        # Exibir recomendações
        for rec in recommendations:
            if rec['type'] == 'success':
                st.success(rec['text'])
            elif rec['type'] == 'warning':
                st.warning(rec['text'])
            else:
                st.info(rec['text'])
    
    def _render_precipitation_analysis(self, df, value_col):
        """Análise específica do impacto da precipitação"""
        
        st.subheader("🌧️ Impacto Detalhado da Precipitação")
        
        if 'precipitacao_total' not in df.columns:
            st.warning("⚠️ Dados de precipitação não disponíveis")
            return
        
        # Categorização por intensidade de chuva
        df_rain = df.copy()
        df_rain['categoria_chuva'] = pd.cut(
            df_rain['precipitacao_total'],
            bins=[-0.1, 0, 2, 10, 25, float('inf')],
            labels=['Sem Chuva', 'Garoa (0-2mm)', 'Chuva Leve (2-10mm)', 'Chuva Moderada (10-25mm)', 'Chuva Intensa (>25mm)']
        )
        
        # Estatísticas por categoria
        rain_stats = df_rain.groupby('categoria_chuva')[value_col].agg(['mean', 'std', 'count']).round(2)
        rain_stats.columns = ['Vendas Médias', 'Desvio Padrão', 'Número de Dias']
        
        st.write("**📊 Vendas por Intensidade de Precipitação:**")
        st.dataframe(rain_stats, use_container_width=True)
        
        # Comparação com/sem chuva
        col1, col2, col3 = st.columns(3)
        
        vendas_sem_chuva = df_rain[df_rain['precipitacao_total'] == 0][value_col].mean()
        vendas_com_chuva = df_rain[df_rain['precipitacao_total'] > 0][value_col].mean()
        
        if vendas_sem_chuva > 0:
            impacto_percentual = ((vendas_com_chuva - vendas_sem_chuva) / vendas_sem_chuva) * 100
        else:
            impacto_percentual = 0
        
        with col1:
            st.metric("☀️ Vendas - Sem Chuva", f"R$ {vendas_sem_chuva:,.2f}".replace(',', '.'))
        
        with col2:
            st.metric("🌧️ Vendas - Com Chuva", f"R$ {vendas_com_chuva:,.2f}".replace(',', '.'))
        
        with col3:
            delta_color = "normal" if abs(impacto_percentual) < 5 else ("inverse" if impacto_percentual < 0 else "normal")
            st.metric("📊 Impacto da Chuva", f"{impacto_percentual:+.1f}%", delta_color=delta_color)
        
        # Gráficos detalhados
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot por categoria de chuva
            fig_rain_box = px.box(
                df_rain.dropna(subset=['categoria_chuva']),
                x='categoria_chuva',
                y=value_col,
                title="Distribuição de Vendas por Intensidade de Chuva",
                labels={'categoria_chuva': 'Categoria de Chuva', value_col: 'Vendas (R$)'}
            )
            fig_rain_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rain_box, use_container_width=True)
        
        with col2:
            # Scatter plot precipitação vs vendas (com escala logarítmica para precipitação)
            df_rain_scatter = df_rain[df_rain['precipitacao_total'] > 0]
            
            if not df_rain_scatter.empty:
                fig_rain_scatter = px.scatter(
                    df_rain_scatter,
                    x='precipitacao_total',
                    y=value_col,
                    title="Relação Precipitação vs Vendas",
                    labels={'precipitacao_total': 'Precipitação (mm)', value_col: 'Vendas (R$)'},
                    log_x=True
                )
                st.plotly_chart(fig_rain_scatter, use_container_width=True)
        
        # Análise mensal da chuva
        if 'mes' in df.columns:
            st.subheader("📅 Padrão Mensal de Chuva e Vendas")
            
            monthly_rain = df_rain.groupby('mes').agg({
                'precipitacao_total': 'mean',
                value_col: 'mean'
            }).round(2)
            
            fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_monthly.add_trace(
                go.Bar(
                    x=list(range(1, 13)),
                    y=monthly_rain['precipitacao_total'],
                    name='Precipitação Média (mm)',
                    opacity=0.7
                ),
                secondary_y=False,
            )
            
            fig_monthly.add_trace(
                go.Scatter(
                    x=list(range(1, 13)),
                    y=monthly_rain[value_col],
                    mode='lines+markers',
                    name='Vendas Médias (R$)',
                    line=dict(color='red', width=3)
                ),
                secondary_y=True,
            )
            
            fig_monthly.update_xaxes(title_text="Mês")
            fig_monthly.update_yaxes(title_text="Precipitação Média (mm)", secondary_y=False)
            fig_monthly.update_yaxes(title_text="Vendas Médias (R$)", secondary_y=True)
            fig_monthly.update_layout(title_text="Padrão Mensal: Chuva vs Vendas")
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Recomendações para precipitação
        self._render_precipitation_recommendations(df_rain, value_col, impacto_percentual)
    
    def _render_precipitation_recommendations(self, df, value_col, impact_percentage):
        """Recomendações baseadas na análise de precipitação"""
        
        st.subheader("💡 Recomendações - Precipitação")
        
        recommendations = []
        
        # Impacto significativo da chuva
        if abs(impact_percentage) > 10:
            if impact_percentage < 0:
                recommendations.append({
                    'type': 'warning',
                    'text': f"📉 **Alto Impacto Negativo** ({impact_percentage:.1f}%): Chuva reduz significativamente as vendas. Desenvolva estratégias para dias chuvosos (delivery, produtos indoor, etc.)."
                })
            else:
                recommendations.append({
                    'type': 'success',
                    'text': f"📈 **Impacto Positivo** ({impact_percentage:.1f}%): Chuva aumenta as vendas! Aproveite dias chuvosos para campanhas especiais."
                })
        
        # Análise de dias consecutivos de chuva
        consecutive_rain_days = 0
        max_consecutive = 0
        
        for _, row in df.iterrows():
            if row['precipitacao_total'] > 0:
                consecutive_rain_days += 1
                max_consecutive = max(max_consecutive, consecutive_rain_days)
            else:
                consecutive_rain_days = 0
        
        if max_consecutive > 3:
            recommendations.append({
                'type': 'info',
                'text': f"🌧️ **Períodos Chuvosos Longos**: Até {max_consecutive} dias consecutivos de chuva detectados. Planeje estratégias para períodos prolongados de mau tempo."
            })
        
        # Análise sazonal de chuva
        if 'estacao' in df.columns:
            rain_by_season = df.groupby('estacao')['precipitacao_total'].mean()
            rainiest_season = rain_by_season.idxmax()
            driest_season = rain_by_season.idxmin()
            
            recommendations.append({
                'type': 'info',
                'text': f"🌦️ **Padrão Sazonal**: {rainiest_season} é a estação mais chuvosa, {driest_season} a mais seca. Ajuste estratégias sazonalmente."
            })
        
        # Exibir recomendações
        for rec in recommendations:
            if rec['type'] == 'success':
                st.success(rec['text'])
            elif rec['type'] == 'warning':
                st.warning(rec['text'])
            else:
                st.info(rec['text'])

# Função para integrar com streamlit_app.py
def show_clima_vendas_page(df, role, store_manager):
    """Função para mostrar a página clima x vendas"""
    
    page = ClimaVendasPage(store_manager)
    page.render()