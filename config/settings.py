# ============================================================================
# ⚙️ config/settings.py - CONFIGURAÇÕES ATUALIZADAS
# ============================================================================

"""Configurações centralizadas para sistema de lojas dinâmico"""

APP_CONFIG = {
    'app_name': 'Sistema Clima & Vendas',
    'app_icon': '🌤️',
    'version': '2.0.0',
    'debug': False,
    
    # Configurações de autenticação
    'auth': {
        'session_timeout_minutes': 30,
        'max_login_attempts': 3,
        'require_strong_password': False
    },
    
    # Configurações de dados
    'data': {
        'datasets_path': 'data/datasets',
        'climate_data_file': 'resumo_diario_climatico.csv',
        'stores_config_file': 'data/stores_config.json',
        'max_file_size_mb': 100,
        'allowed_extensions': ['.csv', '.xlsx', '.xls'],
        
        # Colunas esperadas nos arquivos
        'expected_columns': {
            'date_keywords': ['data', 'date', 'dia', 'fecha'],
            'value_keywords': ['valor', 'value', 'venda', 'sale', 'total', 'receita', 'revenue'],
            'climate_columns': [
                'temp_max', 'temp_min', 'temp_media',
                'umid_max', 'umid_min', 'umid_mediana', 
                'precipitacao_total', 'rad_mediana',
                'vento_vel_media'
            ]
        }
    },
    
    # Configurações de interface
    'ui': {
        'theme': 'light',
        'language': 'pt-BR',
        'items_per_page': 50,
        'default_chart_height': 400,
        'colors': {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db'
        }
    },
    
    # Configurações de validação
    'validation': {
        'max_invalid_dates_percent': 10,  # Máximo 10% de datas inválidas
        'max_invalid_values_percent': 10,  # Máximo 10% de valores inválidos
        'min_records_required': 30,  # Mínimo 30 registros
        'date_range_years': 10  # Máximo 10 anos de diferença
    }
}

# Configurações específicas por ambiente
ENVIRONMENT = 'development'  # development, staging, production

if ENVIRONMENT == 'production':
    APP_CONFIG['debug'] = False
    APP_CONFIG['auth']['require_strong_password'] = True
    APP_CONFIG['auth']['session_timeout_minutes'] = 15  # Mais restritivo em produção
elif ENVIRONMENT == 'development':
    APP_CONFIG['debug'] = True
    APP_CONFIG['auth']['session_timeout_minutes'] = 60  # Mais permissivo em dev
