# ============================================================================
# 🏢 SISTEMA DE ANÁLISE CLIMA & VENDAS
# ============================================================================
# Arquivo Principal - streamlit_app.py
# Responsável apenas por: configuração inicial, autenticação e roteamento
# ============================================================================

import streamlit as st
import sys
from pathlib import Path

# Adicionar diretórios ao path
root_path = Path(__file__).parent
sys.path.append(str(root_path))

# Importações dos módulos
from auth.authentication import AuthenticationManager
from core.router import PageRouter
from core.styles import apply_global_styles
from config.settings import APP_CONFIG

# ============================================================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title=APP_CONFIG['app_name'],
    page_icon=APP_CONFIG['app_icon'],
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': f"# {APP_CONFIG['app_name']}\nVersão {APP_CONFIG['version']}"
    }
)

# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

class ClimaVendasApp:
    """Classe principal da aplicação"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.router = PageRouter()
        
        # Aplicar estilos globais
        apply_global_styles()
    
    def run(self):
        """Executa a aplicação"""
        
        # Verificar autenticação
        if not self.auth_manager.is_authenticated():
            self._show_login_page()
        else:
            self._show_main_application()
    
    def _show_login_page(self):
        """Exibe página de login"""
        from pages.login import LoginPage
        
        login_page = LoginPage(self.auth_manager)
        login_page.render()
    
    def _show_main_application(self):
        """Exibe aplicação principal após login"""
        
        # Verificar timeout de sessão
        if self.auth_manager.check_session_timeout():
            self.auth_manager.logout()
            st.rerun()
        
        # Renderizar aplicação principal
        self.router.render_application(self.auth_manager)

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main():
    """Função principal"""
    try:
        app = ClimaVendasApp()
        app.run()
        
    except Exception as e:
        st.error("❌ Erro interno da aplicação")
        if APP_CONFIG.get('debug', False):
            st.exception(e)

if __name__ == "__main__":
    main()

