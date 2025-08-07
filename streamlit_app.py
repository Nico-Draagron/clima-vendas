# ============================================================================
# 📁 streamlit_app.py - ARQUIVO PRINCIPAL CORRIGIDO
# ============================================================================
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path para importações
root_path = Path(__file__).parent
sys.path.append(str(root_path))

# Configuração da página inicial
st.set_page_config(
    page_title="Dashboard Clima x Vendas",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"  # funçao para esconder menu laterel na tela de login.
)

# CSS profissional para login
st.markdown("""
<style>
    /* Esconder elementos do Streamlit durante login */
    .main > div:first-child {
        padding-top: 0rem;
    }
    
    /* Estilo do container de login */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Header do login */
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-header h1 {
        color: #2c3e50;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .login-header p {
        color: #7f8c8d;
        font-size: 1.1rem;
    }
    
    /* Estilo dos inputs */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #ecf0f1;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Botão de login */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Card de usuários demo */
    .demo-users {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 4px solid #3498db;
    }
    
    .demo-users h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .user-demo {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Esconder elementos desnecessários */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    .stDeployButton {
        visibility: hidden;
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Container principal */
    .main .block-container {
        padding: 3rem 1rem;
    }
    
    /* Mensagens de erro/sucesso */
    .stAlert {
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px;
    }
    
    .spinner {
        width: 30px;
        height: 30px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

class SimpleAuthenticator:
    """Sistema de autenticação simples e robusto"""
    
    def __init__(self):
        self.users = {
            'admin': {
                'password': 'admin123',
                'name': 'Administrador',
                'email': 'admin@empresa.com',
                'role': 'admin'
            },
            'usuario': {
                'password': 'user123', 
                'name': 'Usuário Comum',
                'email': 'usuario@empresa.com',
                'role': 'user'
            },
            'gerente': {
                'password': 'gerente123',
                'name': 'Gerente Regional', 
                'email': 'gerente@empresa.com',
                'role': 'manager'
            }
        }
    
    def authenticate(self, username, password):
        """Autentica usuário"""
        if username in self.users:
            if self.users[username]['password'] == password:
                return True, self.users[username]
        return False, None
    
    def is_logged_in(self):
        """Verifica se usuário está logado"""
        return st.session_state.get('authenticated', False)
    
    def get_user_info(self):
        """Retorna informações do usuário logado"""
        return st.session_state.get('user_info', {})
    
    def login(self, username, user_info):
        """Realiza login do usuário"""
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['user_info'] = user_info
    
    def logout(self):
        """Realiza logout"""
        for key in ['authenticated', 'username', 'user_info']:
            if key in st.session_state:
                del st.session_state[key]

def show_login_page():
    """Tela de login profissional"""
    
    # Container centralizado
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>🌤️ Clima & Vendas</h1>
                <p>Sistema de Análise Preditiva</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Inicializar authenticator
        auth = SimpleAuthenticator()
        
        # Formulário de login
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "👤 Usuário", 
                placeholder="Digite seu usuário",
                key="login_username"
            )
            password = st.text_input(
                "🔒 Senha", 
                type="password",
                placeholder="Digite sua senha",
                key="login_password"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                login_button = st.form_submit_button("🚀 Entrar", use_container_width=True)
            with col_btn2:
                demo_button = st.form_submit_button("👁️ Demo", use_container_width=True)
        
        # Processar login
        if login_button:
            if username and password:
                with st.spinner("Verificando credenciais..."):
                    success, user_info = auth.authenticate(username, password)
                    
                    if success:
                        auth.login(username, user_info)
                        st.success("✅ Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos!")
            else:
                st.error("⚠️ Preencha usuário e senha!")
        
        # Auto-login demo
        if demo_button:
            success, user_info = auth.authenticate('admin', 'admin123')
            auth.login('admin', user_info)
            st.success("✅ Logado como Admin (Demo)")
            st.rerun()
        
        # Card com usuários de demonstração
        st.markdown("""
            <div class="demo-users">
                <h4>👥 Usuários para Teste</h4>
                
                <div class="user-demo">
                    <strong>🔑 Administrador</strong><br>
                    <small>Usuário: <code>admin</code> | Senha: <code>admin123</code></small><br>
                    <em>Acesso total aos dados e funcionalidades</em>
                </div>
                
                <div class="user-demo">
                    <strong>👤 Usuário Comum</strong><br>
                    <small>Usuário: <code>usuario</code> | Senha: <code>user123</code></small><br>
                    <em>Acesso limitado a datasets específicos</em>
                </div>
                
                <div class="user-demo">
                    <strong>👨‍💼 Gerente</strong><br>
                    <small>Usuário: <code>gerente</code> | Senha: <code>gerente123</code></small><br>
                    <em>Acesso a relatórios gerenciais</em>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: white;">
        <p>🔐 Sistema Seguro | 🌟 Desenvolvido com Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def show_main_app():
    """Aplicação principal após login"""
    
    auth = SimpleAuthenticator()
    user_info = auth.get_user_info()
    username = st.session_state.get('username', '')
    
    # Reconfigurar página para app principal
    st.markdown("""
    <style>
        .stApp {
            background: #ffffff;
        }
        
        .main .block-container {
            padding: 1rem;
        }
        
        #MainMenu, footer, header {
            visibility: visible;
        }
        
        .stDeployButton {
            visibility: visible;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header da aplicação
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("# 🌤️ Dashboard Clima x Vendas")
        st.markdown("**Sistema de Análise Preditiva**")
    
    with col2:
        # Informações do usuário
        role_colors = {
            'admin': '#e74c3c',
            'manager': '#f39c12', 
            'user': '#27ae60'
        }
        role_icons = {
            'admin': '👑',
            'manager': '👨‍💼',
            'user': '👤'
        }
        
        role = user_info.get('role', 'user')
        color = role_colors.get(role, '#95a5a6')
        icon = role_icons.get(role, '👤')
        
        st.markdown(f"""
        <div style="
            background: {color}; 
            color: white; 
            padding: 0.5rem 1rem; 
            border-radius: 10px; 
            text-align: center;
            margin: 1rem 0;
        ">
            <strong>{icon} {user_info.get('name', 'Usuário')}</strong><br>
            <small>{role.upper()}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout()
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar para navegação
    with st.sidebar:
        st.markdown("### 🧭 Navegação")
        
        # Opções baseadas no role
        options = ["📊 Dashboard"]
        
        if role in ['admin', 'manager', 'user']:
            options.extend(["🌤️ Clima x Vendas", "📈 Série Temporal"])
        
        if role in ['admin', 'manager']:
            options.append("🤖 Modelo Preditivo")
        
        if role == 'admin':
            options.append("⚙️ Administração")
        
        selected_page = st.radio("Escolha uma opção:", options)
        
        st.markdown("---")
        
        # Informações do sistema
        st.markdown("### ℹ️ Sistema")
        st.info(f"""
        **Usuário:** {username}  
        **Role:** {role.upper()}  
        **Sessão:** Ativa
        """)
    
    # Conteúdo principal baseado na seleção
    if selected_page == "📊 Dashboard":
        show_dashboard(user_info)
    elif selected_page == "🌤️ Clima x Vendas":
        show_clima_vendas(user_info)
    elif selected_page == "📈 Série Temporal":
        show_serie_temporal(user_info)
    elif selected_page == "🤖 Modelo Preditivo":
        show_modelo_preditivo(user_info)
    elif selected_page == "⚙️ Administração":
        show_admin_panel(user_info)

def show_dashboard(user_info):
    """Dashboard principal"""
    st.header("📊 Dashboard Principal")
    
    role = user_info.get('role', 'user')
    
    # Métricas simuladas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Vendas Totais", "R$ 1.234.567", "+12%")
    
    with col2:
        st.metric("📅 Período", "365 dias", "")
    
    with col3:
        if role == 'admin':
            st.metric("💵 Lucro Líquido", "R$ 234.567", "+8%")
        else:
            st.metric("📊 Média Diária", "R$ 3.382", "+5%")
    
    with col4:
        st.metric("🌧️ Dias de Chuva", "127 dias", "-3%")
    
    # Gráfico simples
    st.subheader("📈 Evolução das Vendas")
    
    # Dados simulados
    import numpy as np
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    values = np.random.normal(3000, 500, 30)
    df_demo = pd.DataFrame({'Data': dates, 'Vendas': values})
    
    st.line_chart(df_demo.set_index('Data'))
    
    # Informações específicas por role
    if role == 'admin':
        st.subheader("🔒 Informações Confidenciais")
        st.success("Você tem acesso completo a todos os dados financeiros.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💎 Maior Venda", "R$ 45.678")
        with col2:
            st.metric("📉 Menor Venda", "R$ 1.234")
    
    elif role == 'manager':
        st.subheader("👨‍💼 Visão Gerencial")
        st.info("Acesso a relatórios regionais e métricas de equipe.")
    
    else:
        st.subheader("👤 Resumo Pessoal")
        st.info("Dados agregados e relatórios básicos disponíveis.")

def show_clima_vendas(user_info):
    """Análise clima x vendas"""
    st.header("🌤️ Análise Clima x Vendas")
    st.info("🚧 Funcionalidade será implementada na próxima etapa")
    
    st.markdown("""
    ### Próximas implementações:
    - 📊 Correlação entre temperatura e vendas
    - 🌧️ Impacto da precipitação nos resultados
    - 📈 Gráficos interativos com Plotly
    - 🎯 Insights automáticos
    """)

def show_serie_temporal(user_info):
    """Análise de série temporal"""
    st.header("📈 Análise de Série Temporal")
    st.info("🚧 Funcionalidade será implementada na próxima etapa")

def show_modelo_preditivo(user_info):
    """Modelo preditivo"""
    st.header("🤖 Modelo Preditivo")
    st.info("🚧 Funcionalidade será implementada na próxima etapa")

def show_admin_panel(user_info):
    """Painel administrativo"""
    st.header("⚙️ Painel Administrativo")
    
    tab1, tab2, tab3 = st.tabs(["👥 Usuários", "📁 Datasets", "📊 Logs"])
    
    with tab1:
        st.subheader("Gerenciamento de Usuários")
        
        # Mostrar usuários cadastrados
        auth = SimpleAuthenticator()
        
        st.write("**Usuários Cadastrados:**")
        for username, info in auth.users.items():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"👤 **{info['name']}**")
            with col2:
                st.write(f"Role: `{info['role']}`")
            with col3:
                if st.button("✏️", key=f"edit_{username}"):
                    st.info(f"Editar {info['name']}")
    
    with tab2:
        st.subheader("Gerenciamento de Datasets")
        st.info("Carregar e gerenciar arquivos CSV")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader("📁 Upload de Dataset", type=['csv'])
        if uploaded_file:
            st.success(f"Arquivo {uploaded_file.name} carregado!")
    
    with tab3:
        st.subheader("Logs do Sistema")
        st.code("""
        [2025-01-08 10:30:15] admin: Login realizado
        [2025-01-08 10:31:22] admin: Acessou dashboard
        [2025-01-08 10:35:45] usuario: Login realizado
        [2025-01-08 10:36:12] usuario: Acessou clima_vendas
        """)

def main():
    """Função principal"""
    
    # Inicializar authenticator
    auth = SimpleAuthenticator()
    
    # Verificar se está logado
    if auth.is_logged_in():
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()