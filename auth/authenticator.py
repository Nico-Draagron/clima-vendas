# ============================================================================
# 🔐 auth/authentication.py - GERENCIADOR DE AUTENTICAÇÃO LIMPO
# ============================================================================

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import hashlib

class AuthenticationManager:
    """Gerenciador de autenticação profissional"""
    
    def __init__(self):
        self.session_timeout_minutes = 30
        self.users_db = {
            'admin': {
                'password_hash': self._hash_password('admin123'),
                'name': 'Administrador do Sistema',
                'email': 'admin@empresa.com',
                'role': 'admin',
                'companies': ['all'],  # Acesso a todas as empresas
                'permissions': ['full_access']
            },
            'usuario': {
                'password_hash': self._hash_password('user123'),
                'name': 'Usuário Padrão',
                'email': 'usuario@empresa.com', 
                'role': 'user',
                'companies': ['empresa_001'],  # Acesso limitado
                'permissions': ['read_only']
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Gera hash da senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Autentica usuário"""
        
        if username in self.users_db:
            user_data = self.users_db[username]
            password_hash = self._hash_password(password)
            
            if password_hash == user_data['password_hash']:
                return True, user_data
        
        return False, None
    
    def login(self, username: str, user_data: Dict) -> None:
        """Realiza login do usuário"""
        st.session_state.update({
            'authenticated': True,
            'username': username,
            'user_data': user_data,
            'login_time': datetime.now(),
            'last_activity': datetime.now()
        })
    
    def logout(self) -> None:
        """Realiza logout"""
        keys_to_clear = [
            'authenticated', 'username', 'user_data', 
            'login_time', 'last_activity'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """Verifica se usuário está autenticado"""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> Optional[Dict]:
        """Retorna dados do usuário atual"""
        return st.session_state.get('user_data')
    
    def get_username(self) -> Optional[str]:
        """Retorna username atual"""
        return st.session_state.get('username')
    
    def check_session_timeout(self) -> bool:
        """Verifica timeout da sessão"""
        if 'last_activity' in st.session_state:
            last_activity = st.session_state['last_activity']
            timeout = timedelta(minutes=self.session_timeout_minutes)
            
            if datetime.now() - last_activity > timeout:
                return True
        
        # Atualizar última atividade
        st.session_state['last_activity'] = datetime.now()
        return False
    
    def has_permission(self, permission: str) -> bool:
        """Verifica se usuário tem permissão específica"""
        user_data = self.get_current_user()
        if not user_data:
            return False
        
        user_permissions = user_data.get('permissions', [])
        return permission in user_permissions or 'full_access' in user_permissions
    
    def get_accessible_companies(self) -> list:
        """Retorna empresas acessíveis ao usuário"""
        user_data = self.get_current_user()
        if not user_data:
            return []
        
        companies = user_data.get('companies', [])
        
        if 'all' in companies:
            # Retornar todas as empresas disponíveis
            return self._get_all_companies()
        else:
            return companies
    
    def _get_all_companies(self) -> list:
        """Retorna lista de todas as empresas disponíveis"""
        return [
            'empresa_001',
            'empresa_002', 
            'empresa_003',
            'empresa_004',
            'empresa_005'
        ]