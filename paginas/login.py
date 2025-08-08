# ============================================================================
# 📄 pages/login.py - PÁGINA DE LOGIN LIMPA (SEM DEMO)
# ============================================================================

import streamlit as st
from typing import Optional

class LoginPage:
    """Página de login profissional sem modo demo"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
    
    def render(self):
        """Renderiza página de login"""
        
        # CSS específico para login
        self._apply_login_styles()
        
        # Container centralizado
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            self._render_login_form()
    
    def _apply_login_styles(self):
        """Aplica estilos CSS específicos do login"""
        st.markdown("""
        <style>
            /* Background específico do login */
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            
            /* Container do login */
            .login-container {
                background: white;
                padding: 3rem 2rem;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                margin: 2rem auto;
                max-width: 400px;
            }
            
            /* Header do login */
            .login-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .login-header h1 {
                color: #2c3e50;
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                font-weight: 700;
            }
            
            .login-header p {
                color: #7f8c8d;
                font-size: 1.1rem;
                margin: 0;
            }
            
            /* Inputs do login */
            .stTextInput > div > div > input {
                border-radius: 10px;
                border: 2px solid #ecf0f1;
                padding: 15px;
                font-size: 16px;
                transition: all 0.3s ease;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Botão de login */
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s ease;
                margin-top: 1rem;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            /* Esconder elementos do Streamlit */
            #MainMenu, footer, header, .stDeployButton {
                visibility: hidden;
            }
            
            /* Alertas */
            .stAlert {
                border-radius: 10px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_login_form(self):
        """Renderiza formulário de login"""
        
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>🌤️ Clima & Vendas</h1>
                <p>Sistema de Análise Preditiva</p>
            </div>
        """, unsafe_allow_html=True)
        
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
            
            login_button = st.form_submit_button("🚀 Entrar")
        
        # Processar login
        if login_button:
            if username and password:
                with st.spinner("🔍 Verificando credenciais..."):
                    success, user_data = self.auth_manager.authenticate(username, password)
                    
                    if success:
                        self.auth_manager.login(username, user_data)
                        st.success("✅ Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos!")
            else:
                st.error("⚠️ Preencha todos os campos!")
        
        # Informações de acesso
        st.markdown("""
        <div style="
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
            border-left: 4px solid #667eea;
        ">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">🔑 Credenciais de Acesso</h4>
            
            <div style="
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid #e0e0e0;
            ">
                <strong>👑 Administrador</strong><br>
                <small>Usuário: <code>admin</code> | Senha: <code>admin123</code></small><br>
                <em>Acesso completo: gerenciar lojas, upload de dados, análises</em>
            </div>
            
            <div style="
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid #e0e0e0;
            ">
                <strong>👤 Usuário</strong><br>
                <small>Usuário: <code>usuario</code> | Senha: <code>user123</code></small><br>
                <em>Acesso limitado: visualizar dados e análises das lojas</em>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem; color: white;">
            <p>🔐 Acesso Seguro | 🏢 Sistema Empresarial</p>
        </div>
        """, unsafe_allow_html=True)
