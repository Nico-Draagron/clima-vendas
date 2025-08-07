# ============================================================================
# 🎨 core/styles.py - ESTILOS GLOBAIS
# ============================================================================

import streamlit as st

def apply_global_styles():
    """Aplica estilos CSS globais da aplicação"""
    
    st.markdown("""
    <style>
        /* Variáveis CSS */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --text-color: #2c3e50;
            --background-color: #ffffff;
            --light-background: #f8f9fa;
            --border-color: #e0e6ed;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        /* Reset e configurações globais */
        .main > div:first-child {
            padding-top: 1rem;
        }
        
        /* Esconder elementos do Streamlit */
        #MainMenu, footer, .stDeployButton {
            visibility: hidden;
        }
        
        header[data-testid="stHeader"] {
            height: 0;
        }
        
        /* Container principal */
        .block-container {
            padding: 1rem;
            max-width: 1200px;
        }
        
        /* Títulos */
        h1, h2, h3 {
            color: var(--text-color);
            font-weight: 600;
        }
        
        /* Cards */
        .custom-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            margin: 1rem 0;
        }
        
        /* Métricas */
        [data-testid="metric-container"] {
            background: white;
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 8px;
            box-shadow: var(--shadow);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--light-background);
        }
        
        /* Botões */
        .stButton > button {
            border-radius: 8px;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Alertas */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid var(--border-color);
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        
        /* Dataframes */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }
        
        /* Gráficos */
        .js-plotly-plot {
            border-radius: 8px;
            box-shadow: var(--shadow);
        }
    </style>
    """, unsafe_allow_html=True)


