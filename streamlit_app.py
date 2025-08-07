# ============================================================================
# 🎯 streamlit_app.py - ARQUIVO PRINCIPAL INTEGRADO
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🌤️ Clima x Vendas - Sistema Preditivo",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar diretório atual ao path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Imports dos módulos do sistema
try:
    from auth.auth_system import SimpleAuthenticator
    from data.store_manager import StoreDataManager
    from pages.admin import AdminPage
    from pages.dashboard_preditivo import add_prediction_widgets_to_dashboard
    
    # Imports das páginas principais
    from pages.clima_vendas import show_clima_vendas_page
    from pages.modelo_preditivo import show_modelo_preditivo_page
    from pages.serie_temporal import show_serie_temporal_page
    from pages.previsao_climatica import show_previsao_climatica_page
    
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.info("ℹ️ Verifique se todos os arquivos estão no local correto")
    IMPORTS_OK = False

# ============================================================================
# 🔧 INICIALIZAÇÃO DO SISTEMA
# ============================================================================

def initialize_system():
    """Inicializa componentes do sistema"""
    
    if not IMPORTS_OK:
        return None, None
    
    # Inicializar authenticator
    auth = SimpleAuthenticator()
    
    # Inicializar store manager
    store_manager = StoreDataManager()
    
    return auth, store_manager

# ============================================================================
# 🔐 SISTEMA DE AUTENTICAÇÃO
# ============================================================================

def show_login_page():
    """Página de login"""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🌤️ Sistema Clima x Vendas</h1>
        <h3>Sistema Inteligente de Análise Preditiva</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("👤 Usuário")
            password = st.text_input("🔒 Senha", type="password")
            submit_button = st.form_submit_button("🚀 Entrar", type="primary")
            
            if submit_button:
                auth = st.session_state.get('authenticator')
                if auth and auth.login(username, password):
                    st.success("✅ Login realizado com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Usuário ou senha incorretos")
        
        # Informações de login para demo
        st.markdown("---")
        st.markdown("### 🧪 Demo - Credenciais de Teste")
        
        col_admin, col_user = st.columns(2)
        
        with col_admin:
            st.info("""
            **👑 Administrador:**
            - Usuário: `admin`
            - Senha: `admin123`
            - Acesso completo
            """)
        
        with col_user:
            st.info("""
            **👤 Usuário:**
            - Usuário: `usuario`
            - Senha: `user123`
            - Acesso básico
            """)

# ============================================================================
# 📊 DASHBOARD PRINCIPAL
# ============================================================================

def show_dashboard_page(df, role, store_manager, auth_manager, selected_store_id):
    """Dashboard principal com integração preditiva"""
    
    st.header("📊 Dashboard Principal")
    
    if df is None:
        st.warning("⚠️ Selecione um dataset no menu lateral")
        return
    
    # Verificar coluna de vendas
    stores = store_manager.get_available_stores()
    if selected_store_id not in stores:
        st.error("❌ Loja selecionada não encontrada")
        return
    
    store_info = stores[selected_store_id]
    value_col = store_info['value_column']
    
    if value_col not in df.columns:
        st.error(f"❌ Coluna de vendas '{value_col}' não encontrada nos dados")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_vendas = df[value_col].sum()
        st.metric("💰 Total Vendas", f"R$ {total_vendas:,.2f}".replace(',', '.'))
    
    with col2:
        st.metric("📅 Período", f"{len(df)} dias")
    
    with col3:
        media_vendas = df[value_col].mean()
        st.metric("📊 Média Diária", f"R$ {media_vendas:,.2f}".replace(',', '.'))
    
    with col4:
        if 'precipitacao_total' in df.columns:
            dias_chuva = (df['precipitacao_total'] > 0).sum()
            st.metric("🌧️ Dias com Chuva", f"{dias_chuva}")
        else:
            st.metric("🌧️ Dados de Chuva", "N/A")
    
    # Gráfico de evolução das vendas
    if 'data' in df.columns:
        st.subheader("📈 Evolução das Vendas")
        
        import plotly.express as px
        
        df_plot = df.copy()
        df_plot['data'] = pd.to_datetime(df_plot['data'])
        
        fig = px.line(
            df_plot, 
            x='data', 
            y=value_col,
            title="Faturamento Diário ao Longo do Tempo"
        )
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 🤖 INTEGRAÇÃO DOS WIDGETS PREDITIVOS
    if IMPORTS_OK:
        add_prediction_widgets_to_dashboard(store_manager, auth_manager, selected_store_id)
    
    # Análise rápida por dia da semana
    if 'data' in df.columns:
        st.subheader("📅 Análise por Dia da Semana")
        
        df_analysis = df.copy()
        df_analysis['data'] = pd.to_datetime(df_analysis['data'])
        df_analysis['dia_semana'] = df_analysis['data'].dt.dayofweek
        df_analysis['nome_dia'] = df_analysis['dia_semana'].map({
            0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta',
            4: 'Sexta', 5: 'Sábado', 6: 'Domingo'
        })
        
        weekday_stats = df_analysis.groupby('nome_dia')[value_col].agg(['mean', 'count']).round(2)
        weekday_stats = weekday_stats.reindex([
            'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'
        ])
        weekday_stats.columns = ['Vendas Médias (R$)', 'Número de Dias']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(weekday_stats, use_container_width=True)
        
        with col2:
            fig_weekday = px.bar(
                x=weekday_stats.index,
                y=weekday_stats['Vendas Médias (R$)'],
                title="Vendas Médias por Dia da Semana",
                labels={'x': 'Dia da Semana', 'y': 'Vendas Médias (R$)'}
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Informações específicas por role
    if role == "admin":
        st.subheader("🔒 Informações Confidenciais (Admin Only)")
        st.info("Esta seção só é visível para administradores")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Maior Venda", f"R$ {df[value_col].max():,.2f}".replace(',', '.'))
        with col2:
            st.metric("📉 Menor Venda", f"R$ {df[value_col].min():,.2f}".replace(',', '.'))
        with col3:
            cv = df[value_col].std() / df[value_col].mean()
            st.metric("📊 Coef. Variação", f"{cv:.3f}")

# ============================================================================
# 🎮 INTERFACE PRINCIPAL
# ============================================================================

def show_main_app():
    """Interface principal do aplicativo"""
    
    auth = st.session_state.get('authenticator')
    store_manager = st.session_state.get('store_manager')
    
    if not auth or not store_manager:
        st.error("❌ Erro na inicialização do sistema")
        return
    
    user_info = auth.get_user_info()
    username = user_info.get('name', 'Usuário')
    role = user_info.get('role', 'user')
    
    # Sidebar com navegação
    with st.sidebar:
        st.markdown(f"### 👋 Olá, {username}!")
        st.markdown(f"**Role:** {role.upper()}")
        
        st.markdown("---")
        
        # Seleção de loja
        st.markdown("### 🏪 Seleção de Loja")
        
        stores = store_manager.get_available_stores()
        
        if stores:
            store_options = {f"{info['display_name']} ({store_id})": store_id 
                           for store_id, info in stores.items()}
            
            selected_display = st.selectbox(
                "Escolha uma loja:",
                options=list(store_options.keys()),
                key="store_selector"
            )
            
            selected_store_id = store_options[selected_display]
            
            # Carregar dados da loja selecionada
            with st.spinner("Carregando dados..."):
                df = store_manager.load_store_data(selected_store_id)
            
            if df is not None:
                st.success(f"✅ {len(df)} registros carregados")
                
                # Informações da loja
                store_info = stores[selected_store_id]
                st.markdown(f"**📊 Coluna de Vendas:** {store_info['value_column']}")
                
                if not df.empty:
                    date_range = f"{df['data'].min().strftime('%d/%m/%Y')} - {df['data'].max().strftime('%d/%m/%Y')}"
                    st.markdown(f"**📅 Período:** {date_range}")
            else:
                st.error("❌ Erro ao carregar dados")
                df = None
        else:
            st.warning("⚠️ Nenhuma loja configurada")
            selected_store_id = None
            df = None
        
        st.markdown("---")
        
        # Menu de navegação
        st.markdown("### 📋 Menu de Navegação")
        
        # Opções baseadas no role
        menu_options = [
            "📊 Dashboard",
            "🌤️ Clima x Vendas",
            "📈 Série Temporal",
            "🤖 Modelo Preditivo",
            "🔮 Previsão Climática"
        ]
        
        if role == "admin":
            menu_options.append("⚙️ Administração")
        
        selected_page = st.radio("Escolha uma página:", menu_options)
        
        st.markdown("---")
        
        # Informações do sistema
        st.markdown("### ℹ️ Sistema")
        st.markdown(f"""
        **Versão:** 2.0.0  
        **Última Atualização:** {datetime.now().strftime('%d/%m/%Y')}  
        **Status:** 🟢 Online
        """)
        
        # Logout
        if st.button("🚪 Logout", type="secondary"):
            auth.logout()
            st.rerun()
    
    # Conteúdo principal baseado na seleção
    if selected_page == "📊 Dashboard":
        show_dashboard_page(df, role, store_manager, auth, selected_store_id)
    
    elif selected_page == "🌤️ Clima x Vendas":
        if IMPORTS_OK and df is not None:
            show_clima_vendas_page(df, role, store_manager)
        else:
            st.error("❌ Dados não disponíveis ou módulo não encontrado")
    
    elif selected_page == "📈 Série Temporal":
        if IMPORTS_OK and df is not None:
            show_serie_temporal_page(df, role, store_manager)
        else:
            st.error("❌ Dados não disponíveis ou módulo não encontrado")
    
    elif selected_page == "🤖 Modelo Preditivo":
        if IMPORTS_OK and df is not None:
            show_modelo_preditivo_page(df, role, store_manager, auth)
        else:
            st.error("❌ Dados não disponíveis ou módulo não encontrado")
    
    elif selected_page == "🔮 Previsão Climática":
        if IMPORTS_OK and df is not None:
            show_previsao_climatica_page(df, role, store_manager)
        else:
            st.error("❌ Dados não disponíveis ou módulo não encontrado")
    
    elif selected_page == "⚙️ Administração" and role == "admin":
        if IMPORTS_OK:
            admin_page = AdminPage(auth)
            admin_page.render()
        else:
            st.error("❌ Módulo administrativo não encontrado")

# ============================================================================
# 🎯 FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal da aplicação"""
    
    # Inicializar sistema se não existir na sessão
    if 'authenticator' not in st.session_state or 'store_manager' not in st.session_state:
        auth, store_manager = initialize_system()
        
        if auth is None or store_manager is None:
            st.error("❌ Falha na inicialização do sistema")
            st.stop()
        
        st.session_state['authenticator'] = auth
        st.session_state['store_manager'] = store_manager
    
    # Verificar se usuário está logado
    auth = st.session_state['authenticator']
    
    if auth.is_logged_in():
        show_main_app()
    else:
        show_login_page()

# ============================================================================
# 🚀 PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()