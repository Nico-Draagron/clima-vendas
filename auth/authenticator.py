import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from datetime import datetime, timedelta
import hashlib

class AuthManager:
    def __init__(self):
        self.config_file = 'config/auth_config.yaml'
        self.config = self.load_config()
        self.authenticator = self.setup_authenticator()
    
    def load_config(self):
        """Carrega configuração de autenticação"""
        try:
            # Tenta carregar do arquivo local primeiro
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as file:
                    return yaml.load(file, Loader=SafeLoader)
            else:
                # Se não existir, cria configuração padrão
                return self.create_default_config()
        except Exception as e:
            st.error(f"Erro ao carregar configuração: {e}")
            return self.create_default_config()
    
    def create_default_config(self):
        """Cria configuração padrão de usuários"""
        # Senhas são hasheadas automaticamente
        default_config = {
            'cookie': {
                'expiry_days': 1,  # Expira em 1 dia por segurança
                'key': 'auth_key_12345',  # Alterar em produção
                'name': 'auth_cookie'
            },
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@empresa.com',
                        'first_name': 'Admin',
                        'last_name': 'System',
                        'password': 'admin123',  # Será hasheada
                        'role': 'admin',
                        'datasets': ['all']  # Acesso a todos
                    },
                    'usuario': {
                        'email': 'usuario@empresa.com', 
                        'first_name': 'Usuário',
                        'last_name': 'Comum',
                        'password': 'user123',  # Será hasheada
                        'role': 'user',
                        'datasets': ['Loja1_dados_unificados.csv']  # Acesso limitado
                    }
                }
            }
        }
        
        # Hashear senhas se necessário
        for username, user_data in default_config['credentials']['usernames'].items():
            if isinstance(user_data['password'], str) and len(user_data['password']) < 50:
                # Senha ainda não foi hasheada
                user_data['password'] = stauth.Hasher([user_data['password']]).generate()[0]
        
        # Salvar configuração
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config):
        """Salva configuração no arquivo"""
        try:
            os.makedirs('config', exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            st.error(f"Erro ao salvar configuração: {e}")
    
    def setup_authenticator(self):
        """Configura o autenticador"""
        st.write("CONFIG DEBUG:", self.config)
        try:
            return stauth.Authenticate(
                self.config['credentials'],
                self.config['cookie']['name'],
                self.config['cookie']['key'],
                self.config['cookie']['expiry_days']
            )
        except Exception as e:
            st.error(f"Erro ao configurar autenticador: {e}")
            return None
    
    def authenticate(self):
        """Executa processo de autenticação"""
        if self.authenticator is None:
            return None, None, None, None

        try:
            # Widget de login
            name, authentication_status, username = self.authenticator.login('Login', 'main')

            # Se autenticado, obter role do usuário
            role = None
            if authentication_status and username:
                role = self.get_user_role(username)
                # Salvar informações na sessão
                st.session_state['user_role'] = role
                st.session_state['username'] = username
                st.session_state['name'] = name
                st.session_state['login_time'] = datetime.now()

            return name, authentication_status, username, role

        except Exception as e:
            st.error(f"Erro na autenticação: {e}")
            return None, False, None, None
    
    def get_user_role(self, username):
        """Obtém role do usuário"""
        try:
            return self.config['credentials']['usernames'][username].get('role', 'user')
        except:
            return 'user'
    
    def get_user_datasets(self, username):
        """Obtém datasets permitidos para o usuário"""
        try:
            datasets = self.config['credentials']['usernames'][username].get('datasets', [])
            return datasets
        except:
            return []
    
    def logout(self):
        """Executa logout do usuário"""
        if self.authenticator:
            self.authenticator.logout('Logout', 'sidebar')
        
        # Limpar sessão
        keys_to_remove = ['user_role', 'username', 'name', 'login_time']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    def check_session_timeout(self, timeout_minutes=30):
        """Verifica timeout da sessão"""
        if 'login_time' in st.session_state:
            login_time = st.session_state['login_time']
            if datetime.now() - login_time > timedelta(minutes=timeout_minutes):
                self.logout()
                st.error("⏰ Sessão expirada. Faça login novamente.")
                st.rerun()
