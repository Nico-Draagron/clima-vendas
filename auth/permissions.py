import streamlit as st
import pandas as pd
from typing import List, Dict
import os

class PermissionManager:
    """Gerenciador de permissões baseado em roles"""
    
    def __init__(self):
        self.role_permissions = {
            'admin': {
                'datasets': ['all'],  # Acesso a todos os datasets
                'features': ['dashboard', 'clima_vendas', 'serie_temporal', 'modelo_preditivo', 'admin'],
                'data_access': 'full'  # Acesso completo aos dados
            },
            'user': {
                'datasets': [],  # Configurado por usuário
                'features': ['dashboard', 'clima_vendas', 'serie_temporal'],
                'data_access': 'limited'  # Acesso limitado (dados mascarados)
            }
        }
    
    def get_user_permissions(self, role: str) -> Dict:
        """Retorna permissões do role"""
        return self.role_permissions.get(role, self.role_permissions['user'])
    
    def can_access_feature(self, role: str, feature: str) -> bool:
        """Verifica se role pode acessar funcionalidade"""
        permissions = self.get_user_permissions(role)
        return feature in permissions.get('features', [])
    
    def get_allowed_datasets(self, role: str, username: str = None) -> List[str]:
        """Retorna datasets permitidos para o role/usuário"""
        if role == 'admin':
            return self.list_all_datasets()
        
        # Para usuários comuns, buscar configuração específica
        if username and 'authentication_status' in st.session_state:
            from auth.authenticator import AuthManager
            auth_manager = AuthManager()
            user_datasets = auth_manager.get_user_datasets(username)
            
            if 'all' in user_datasets:
                return self.list_all_datasets()
            else:
                return user_datasets
        
        return []
    
    def list_all_datasets(self) -> List[str]:
        """Lista todos os datasets disponíveis"""
        datasets_path = 'data/datasets'
        if not os.path.exists(datasets_path):
            return []
        
        datasets = []
        for file in os.listdir(datasets_path):
            if file.endswith('.csv'):
                datasets.append(file)
        
        return datasets
    
    def mask_sensitive_data(self, df, role: str):
        """Aplica mascaramento de dados baseado no role"""
        if role == 'admin':
            return df  # Admin vê todos os dados
        
        # Para usuários comuns, mascarar dados sensíveis
        masked_df = df.copy()
        
        # Lista de colunas sensíveis (dados financeiros)
        sensitive_columns = [
            'valor_total', 'valor_loja_01', 'valor_loja_02', 
            'valor_medio', 'receita', 'lucro', 'custos'
        ]
        
        for col in sensitive_columns:
            if col in masked_df.columns:
                # Aplicar mascaramento: mostrar apenas faixas
                masked_df[col] = masked_df[col].apply(self._mask_financial_value)
        
        return masked_df
    
    def _mask_financial_value(self, value):
        """Mascara valores financeiros em faixas"""
        try:
            if pd.isna(value):
                return "N/A"
            
            if value < 1000:
                return "< R$ 1.000"
            elif value < 5000:
                return "R$ 1.000 - R$ 5.000"
            elif value < 10000:
                return "R$ 5.000 - R$ 10.000"
            elif value < 50000:
                return "R$ 10.000 - R$ 50.000"
            else:
                return "> R$ 50.000"
        except:
            return "[PROTEGIDO]"

def check_user_permissions(role: str, username: str = None) -> List[str]:
    """Função auxiliar para verificar permissões de datasets"""
    perm_manager = PermissionManager()
    return perm_manager.get_allowed_datasets(role, username)

def require_role(required_roles: List[str]):
    """Decorator para exigir roles específicos"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_role = st.session_state.get('user_role', 'user')
            
            if current_role not in required_roles:
                st.error(f"❌ Acesso negado. Roles necessários: {', '.join(required_roles)}")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
