# ============================================================================
# 🏪 pages/stores.py - SELEÇÃO E VISUALIZAÇÃO DE LOJAS
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from data.store_manager import StoreDataManager

class StoresPage:
    """Página de seleção e visualização de dados das lojas"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
    
    def render(self):
        """Renderiza página de lojas"""
        
        st.markdown("# 🏪 Gerenciamento de Lojas")
        st.markdown("**Visualize e analise dados específicos de cada loja**")
        
        # Carregar lojas disponíveis
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.warning("⚠️ Nenhuma loja configurada no sistema")
            if self.auth_manager.has_permission('manage_stores'):
                st.info("💡 Use a opção 'Registrar Loja' para adicionar novas lojas")
            return
        
        # === SELEÇÃO DE LOJA ===
        st.subheader("🎯 Selecionar Loja")
        
        # Criar opções de seleção
        store_options = {}
        for store_id, store_info in stores.items():
            display_text = f"{store_info['display_name']} ({store_id})"
            store_options[display_text] = store_id
        
        selected_display = st.selectbox(
            "Escolha uma loja para análise:",
            options=list(store_options.keys()),
            help="Selecione a loja para visualizar dados detalhados"
        )
        
        selected_store_id = store_options[selected_display]
        selected_store_info = stores[selected_store_id]
        
        # === INFORMAÇÕES DA LOJA ===
        self._render_store_info(selected_store_id, selected_store_info)
        
        # === DADOS DA LOJA ===
        df = self.store_manager.load_store_data(selected_store_id)
        
        if df is None or df.empty:
            st.error(f"❌ Não foi possível carregar dados da loja {selected_store_info['display_name']}")
            return
        
        # === FILTROS ===
        df_filtered = self._render_filters(df)
        
        # === VISUALIZAÇÕES ===
        self._render_store_visualizations(df_filtered, selected_store_info)
        
        # === DADOS TABULARES ===
        self._render_data_table(df_filtered, selected_store_info)
    
    def _render_store_info(self, store_id, store_info):
        """Renderiza informações da loja selecionada"""
        
        st.subheader(f"ℹ️ {store_info['display_name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **🏷️ Identificação**
            - **ID:** {store_id}
            - **Nome:** {store_info['name']}
            - **Status:** {store_info.get('status', 'active').upper()}
            """)
        
        with col2:
            st.markdown(f"""
            **📁 Arquivo de Dados**
            - **Arquivo:** {store_info['csv_file']}
            - **Coluna de Valor:** {store_info['value_column']}
            - **Dados Climáticos:** {'✅ Sim' if store_info.get('has_climate_data') else '❌ Não'}
            """)
        
        with col3:
            st.markdown(f"""
            **📅 Informações Adicionais**
            - **Criado em:** {store_info.get('created_date', 'N/A')}
            - **Localização:** {store_info.get('location', 'N/A')}
            - **Descrição:** {store_info.get('description', 'N/A')}
            """)
    
    def _render_filters(self, df):
        """Renderiza filtros para os dados"""
        
        st.subheader("🎛️ Filtros de Dados")
        
        with st.expander("⚙️ Configurar Filtros", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filtro de data
                if 'data' in df.columns:
                    date_min = df['data'].min().date()
                    date_max = df['data'].max().date()
                    
                    start_date = st.date_input(
                        "📅 Data Inicial",
                        value=date_min,
                        min_value=date_min,
                        max_value=date_max
                    )
                    
                    end_date = st.date_input(
                        "📅 Data Final", 
                        value=date_max,
                        min_value=date_min,
                        max_value=date_max
                    )
                    
                    # Aplicar filtro de data
                    df = df[
                        (df['data'] >= pd.to_datetime(start_date)) &
                        (df['data'] <= pd.to_datetime(end_date))
                    ]
            
            with col2:
                # Filtro de temperatura
                if 'temp_media' in df.columns:
                    temp_range = st.slider(
                        "🌡️ Faixa de Temperatura (°C)",
                        min_value=float(df['temp_media'].min()),
                        max_value=float(df['temp_media'].max()),
                        value=(float(df['temp_media'].min()), float(df['temp_media'].max())),
                        step=0.1
                    )
                    
                    df = df[
                        (df['temp_media'] >= temp_range[0]) &
                        (df['temp_media'] <= temp_range[1])
                    ]
            
            with col3:
                # Filtro de precipitação
                if 'precipitacao_total' in df.columns:
                    rain_filter = st.selectbox(
                        "🌧️ Filtro de Chuva",
                        options=['Todos os dias', 'Apenas com chuva', 'Apenas sem chuva'],
                        index=0
                    )
                    
                    if rain_filter == 'Apenas com chuva':
                        df = df[df['precipitacao_total'] > 0]
                    elif rain_filter == 'Apenas sem chuva':
                        df = df[df['precipitacao_total'] == 0]
        
        st.info(f"📊 Dados filtrados: {len(df)} registros de {len(self.store_manager.load_store_data('loja_001') or [])} totais")
        
        return df
    
    def _render_store_visualizations(self, df, store_info):
        """Renderiza visualizações da loja"""
        
        st.subheader("📈 Visualizações")
        
        value_col = store_info['value_column']
        
        # Verificar se coluna de valor existe
        if value_col not in df.columns:
            st.error(f"❌ Coluna de valor '{value_col}' não encontrada nos dados")
            return
        
        # Tabs de visualização
        tab1, tab2, tab3 = st.tabs(["📊 Vendas no Tempo", "🌡️ Clima vs Vendas", "📈 Distribuições"])
        
        with tab1:
            # Gráfico de vendas ao longo do tempo
            if 'data' in df.columns:
                fig = px.line(
                    df,
                    x='data',
                    y=value_col,
                    title=f"Evolução das Vendas - {store_info['display_name']}",
                    labels={'data': 'Data', value_col: 'Vendas (R$)'}
                )
                
                fig.update_layout(
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Média móvel
                if len(df) > 7:
                    window = st.slider("📊 Janela da Média Móvel (dias)", 3, 30, 7)
                    df_ma = df.copy()
                    df_ma[f'{value_col}_ma'] = df_ma[value_col].rolling(window=window).mean()
                    
                    fig_ma = px.line(
                        df_ma,
                        x='data',
                        y=[value_col, f'{value_col}_ma'],
                        title=f"Vendas com Média Móvel ({window} dias)",
                        labels={'data': 'Data', 'value': 'Vendas (R$)'}
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
        
        with tab2:
            # Correlação clima vs vendas
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temp_media' in df.columns:
                    fig_temp = px.scatter(
                        df,
                        x='temp_media',
                        y=value_col,
                        title="Temperatura vs Vendas",
                        labels={'temp_media': 'Temperatura Média (°C)', value_col: 'Vendas (R$)'},
                        trendline="ols"
                    )
                    
                    st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                if 'precipitacao_total' in df.columns:
                    fig_rain = px.scatter(
                        df,
                        x='precipitacao_total',
                        y=value_col,
                        title="Precipitação vs Vendas",
                        labels={'precipitacao_total': 'Precipitação (mm)', value_col: 'Vendas (R$)'},
                        trendline="ols"
                    )
                    
                    st.plotly_chart(fig_rain, use_container_width=True)
        
        with tab3:
            # Distribuições
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma de vendas
                fig_hist = px.histogram(
                    df,
                    x=value_col,
                    nbins=30,
                    title="Distribuição das Vendas",
                    labels={value_col: 'Vendas (R$)', 'count': 'Frequência'}
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot por dia da semana
                if 'data' in df.columns:
                    df_box = df.copy()
                    df_box['dia_semana'] = df_box['data'].dt.day_name()
                    
                    fig_box = px.box(
                        df_box,
                        x='dia_semana',
                        y=value_col,
                        title="Vendas por Dia da Semana",
                        labels={'dia_semana': 'Dia da Semana', value_col: 'Vendas (R$)'}
                    )
                    
                    fig_box.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
    
    def _render_data_table(self, df, store_info):
        """Renderiza tabela de dados"""
        
        st.subheader("📋 Dados Tabulares")
        
        # Configurações da tabela
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all = st.checkbox("📄 Mostrar todos os dados", value=False)
        
        with col2:
            if not show_all:
                num_rows = st.selectbox("🔢 Número de linhas", [10, 25, 50, 100], index=1)
            else:
                num_rows = len(df)
        
        with col3:
            # Download dos dados
            csv_data = df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv_data,
                file_name=f"{store_info['display_name']}_dados_filtrados.csv",
                mime="text/csv"
            )
        
        # Exibir tabela
        if show_all:
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.dataframe(df.head(num_rows), use_container_width=True)
        
        # Estatísticas resumidas
        if st.checkbox("📊 Mostrar estatísticas resumidas"):
            st.subheader("📈 Estatísticas Descritivas")
            
            # Selecionar apenas colunas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("ℹ️ Nenhuma coluna numérica encontrada para estatísticas")
