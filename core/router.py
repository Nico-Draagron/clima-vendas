
# ============================================================================
# 🧭 core/router.py - ROTEADOR DE PÁGINAS
# ============================================================================


APP_CONFIG = {
    'debug': False
}

import streamlit as st
from typing import Dict, Any

class PageRouter:
    """Gerenciador de roteamento e navegação"""
    
    def __init__(self):
        self.pages = self._initialize_pages()
    
    def _initialize_pages(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa configuração das páginas"""
        return {
            'home': {
                'title': '🏠 Início',
                'module': 'pages.home',
                'class': 'HomePage',
                'roles': ['admin', 'user'],
                'order': 1
            },
            'datasets': {
                'title': '📊 Datasets',
                'module': 'pages.datasets', 
                'class': 'DatasetsPage',
                'roles': ['admin', 'user'],
                'order': 2
            },
            'climate_prediction': {
                'title': '🌤️ Previsão Climática',
                'module': 'pages.climate_prediction',
                'class': 'ClimatePredictionPage', 
                'roles': ['admin', 'user'],
                'order': 3
            },
            'analytics': {
                'title': '📈 Análises',
                'module': 'pages.analytics',
                'class': 'AnalyticsPage',
                'roles': ['admin', 'user'], 
                'order': 4
            },
            'admin': {
                'title': '⚙️ Administração',
                'module': 'pages.admin',
                'class': 'AdminPage',
                'roles': ['admin'],
                'order': 5
            }
        }
    
    def get_available_pages(self, user_role: str) -> Dict[str, Dict[str, Any]]:
        """Retorna páginas disponíveis para o role do usuário"""
        available = {}
        
        for page_id, page_config in self.pages.items():
            if user_role in page_config['roles']:
                available[page_id] = page_config
        
        # Ordenar páginas
        return dict(sorted(available.items(), key=lambda x: x[1]['order']))
    
    def render_application(self, auth_manager):
        """Renderiza aplicação principal com navegação"""
        user_data = auth_manager.get_current_user()
        user_role = user_data.get('role', 'user')
        
        # Header da aplicação
        self._render_header(auth_manager)
        
        # Navegação
        selected_page = self._render_navigation(user_role)
        
        # Renderizar página selecionada
        self._render_page(selected_page, auth_manager)
    
    def _render_header(self, auth_manager):
        """Renderiza header da aplicação"""
        user_data = auth_manager.get_current_user()
        username = auth_manager.get_username()
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown("# 🌤️ Sistema Clima & Vendas")
            st.markdown("**Análise Preditiva e Business Intelligence**")
        
        with col2:
            # Badge do usuário
            role = user_data.get('role', 'user')
            role_colors = {
                'admin': '#e74c3c',
                'user': '#27ae60'
            }
            
            st.markdown(f"""
            <div style="
                background: {role_colors.get(role, '#95a5a6')};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                text-align: center;
                font-weight: 600;
            ">
                {user_data.get('name', 'Usuário')}
                <br><small>{role.upper()}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                auth_manager.logout()
                st.rerun()
        
        st.markdown("---")
    
    def _render_navigation(self, user_role: str) -> str:
        """Renderiza navegação lateral"""
        available_pages = self.get_available_pages(user_role)
        
        with st.sidebar:
            st.markdown("### 🧭 Navegação")
            
            # Opções de navegação
            page_options = [page_config['title'] for page_config in available_pages.values()]
            page_ids = list(available_pages.keys())
            
            selected_index = st.radio(
                "Escolha uma página:",
                range(len(page_options)),
                format_func=lambda x: page_options[x],
                key="page_navigation"
            )
            
            return page_ids[selected_index]
    
    def _render_page(self, page_id: str, auth_manager):
        """Renderiza página selecionada"""
        if page_id not in self.pages:
            st.error(f"❌ Página '{page_id}' não encontrada")
            return
        
        page_config = self.pages[page_id]
        
        try:
            # Importar e instanciar página dinamicamente
            module_name = page_config['module']
            class_name = page_config['class']
            
            module = __import__(module_name, fromlist=[class_name])
            page_class = getattr(module, class_name)
            
            # Instanciar e renderizar página
            page_instance = page_class(auth_manager)
            page_instance.render()
            
        except ImportError:
            st.warning(f"⚠️ Página '{page_config['title']}' em desenvolvimento")
            st.info("Esta funcionalidade será implementada em breve.")
        except Exception as e:
            st.error(f"❌ Erro ao carregar página: {e}")
            if APP_CONFIG.get('debug', False):
                st.exception(e)