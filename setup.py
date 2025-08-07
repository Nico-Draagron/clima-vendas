# ============================================================================
# 🚀 setup.py - SCRIPT DE INICIALIZAÇÃO DO SISTEMA
# ============================================================================

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def create_directory_structure():
    """Cria estrutura de diretórios necessária"""
    
    print("📁 Criando estrutura de diretórios...")
    
    directories = [
        'auth',
        'data',
        'data/datasets',
        'pages',
        'utils',
        '.streamlit'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")
    
    # Criar arquivos __init__.py
    init_files = [
        'auth/__init__.py',
        'data/__init__.py', 
        'pages/__init__.py',
        'utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# -*- coding: utf-8 -*-\n')
            print(f"✅ {init_file}")

def create_sample_data():
    """Cria dados de exemplo para teste"""
    
    print("\n📊 Criando dados de exemplo...")
    
    # Gerar dados sintéticos
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start_date, periods=365, freq='D')
    
    np.random.seed(42)  # Para reproducibilidade
    
    data = []
    for i, date in enumerate(dates):
        # Padrões sazonais
        dia_ano = date.dayofyear
        temp_base = 25 + 10 * np.sin(2 * np.pi * dia_ano / 365.25)
        
        # Temperatura
        temp_media = temp_base + np.random.normal(0, 3)
        temp_max = temp_media + np.random.uniform(3, 7)
        temp_min = temp_media - np.random.uniform(2, 5)
        
        # Precipitação
        precipitacao = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
        
        # Umidade
        umidade = np.random.uniform(40, 90) + (precipitacao * 2)
        umidade = min(100, umidade)
        
        # Radiação
        radiacao = max(10, 25 + 10 * np.sin(2 * np.pi * dia_ano / 365.25) + np.random.normal(0, 5))
        
        # Vento
        vento_vel = max(0, np.random.normal(15, 8))
        vento_raj = vento_vel + np.random.uniform(5, 15)
        
        # Vendas (correlacionadas com clima)
        vendas_base = 50000
        
        # Efeito da temperatura
        if 20 <= temp_media <= 28:
            vendas_base *= 1.1
        elif temp_media > 32 or temp_media < 15:
            vendas_base *= 0.9
        
        # Efeito da chuva
        if precipitacao > 10:
            vendas_base *= 0.8
        elif 1 < precipitacao <= 10:
            vendas_base *= 0.95
        
        # Efeito do dia da semana
        if date.weekday() >= 5:  # Fins de semana
            vendas_base *= 1.2
        
        # Adicionar ruído
        vendas_final = vendas_base * (1 + np.random.normal(0, 0.15))
        vendas_final = max(0, vendas_final)
        
        # Percentuais de dados faltantes (simulados)
        temp_faltantes = "0.0%" if np.random.random() > 0.05 else f"{np.random.uniform(0.1, 5.0):.1f}%"
        umidade_faltantes = "0.0%" if np.random.random() > 0.03 else f"{np.random.uniform(0.1, 3.0):.1f}%"
        radiacao_faltantes = "0.0%" if np.random.random() > 0.07 else f"{np.random.uniform(0.1, 7.0):.1f}%"
        vento_faltantes = "0.0%" if np.random.random() > 0.04 else f"{np.random.uniform(0.1, 4.0):.1f}%"
        precipitacao_faltantes = "0.0%" if np.random.random() > 0.02 else f"{np.random.uniform(0.1, 2.0):.1f}%"
        
        data.append({
            'data': date.strftime('%Y-%m-%d'),
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'temp_media': round(temp_media, 1),
            'umid_max': round(min(100, umidade + np.random.uniform(0, 10)), 1),
            'umid_min': round(max(0, umidade - np.random.uniform(0, 10)), 1),
            'umid_mediana': round(umidade, 1),
            'rad_min': round(max(0, radiacao - np.random.uniform(5, 10)), 1),
            'rad_max': round(radiacao + np.random.uniform(5, 15), 1),
            'rad_mediana': round(radiacao, 1),
            'vento_raj_max': round(vento_raj, 1),
            'vento_vel_media': round(vento_vel, 1),
            'precipitacao_total': round(precipitacao, 1),
            'temp_inst_faltantes_pct': temp_faltantes,
            'umidade_faltantes_pct': umidade_faltantes,
            'radiacao_faltantes_pct': radiacao_faltantes,
            'vento_vel_faltantes_pct': vento_faltantes,
            'precipitacao_faltantes_pct': precipitacao_faltantes,
            'valor_loja_01': round(vendas_final, 2)
        })
    
    # Salvar CSV
    df = pd.DataFrame(data)
    output_file = 'data/datasets/Loja1_dados_unificados.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Dados de exemplo criados: {output_file}")
    print(f"   📊 {len(df)} registros de {df['data'].min()} a {df['data'].max()}")
    print(f"   📈 Vendas de R$ {df['valor_loja_01'].min():,.2f} a R$ {df['valor_loja_01'].max():,.2f}")

def create_store_config():
    """Cria configuração inicial das lojas"""
    
    print("\n🏪 Criando configuração das lojas...")
    
    stores_config = {
        "loja_001": {
            "display_name": "Loja Principal",
            "csv_file": "Loja1_dados_unificados.csv",
            "value_column": "valor_loja_01",
            "location": "Agudo, RS",
            "description": "Loja principal com dados históricos completos",
            "status": "active",
            "created_date": datetime.now().strftime('%Y-%m-%d'),
            "data_source": "sistema_interno"
        }
    }
    
    # Criar diretório se não existir
    os.makedirs('data', exist_ok=True)
    
    # Salvar configuração
    config_file = 'data/stores_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(stores_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuração das lojas criada: {config_file}")

def create_streamlit_config():
    """Cria configuração do Streamlit"""
    
    print("\n⚙️ Criando configuração do Streamlit...")
    
    config_content = """[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
showErrorDetails = true

[client]
caching = true
displayEnabled = true

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
"""
    
    config_file = '.streamlit/config.toml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Configuração do Streamlit criada: {config_file}")

def create_gitignore():
    """Cria arquivo .gitignore"""
    
    print("\n📝 Criando .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Streamlit
.streamlit/secrets.toml

# Data (opcional - descomente se não quiser versionar dados)
# data/datasets/*.csv

# Modelos treinados
*.pkl
*.joblib
*.h5

# Temporary files
*.tmp
*.temp

# API Keys (se houver)
.env
api_keys.txt
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore criado")

def create_readme():
    """Cria README.md atualizado"""
    
    print("\n📖 Criando README.md...")
    
    readme_content = """# 🌤️ Sistema Clima x Vendas

Sistema inteligente de análise preditiva para correlação entre clima e vendas.

## 🚀 Funcionalidades

### ✅ Implementado (Etapa 1 + 2)
- 📊 **Dashboard Principal** com widgets preditivos
- 🌤️ **Análise Clima x Vendas** com correlações e insights
- 📈 **Série Temporal** completa (decomposição, autocorrelação)
- 🤖 **Modelo Preditivo** integrado (Random Forest, Bootstrap)
- 🔮 **Previsão Climática** com alertas automáticos
- ⚙️ **Painel Administrativo** completo
- 🔐 **Sistema de Autenticação** com roles

### 🔧 Próximas Etapas
- 📊 Análises estatísticas avançadas (ANOVA, testes de hipótese)
- 🧠 Machine Learning avançado (XGBoost, LSTM, SHAP)
- 📋 Dashboards especializados

## 🛠️ Instalação

1. **Clone o repositório**
```bash
git clone [url-do-repositorio]
cd clima-vendas
```

2. **Execute o setup automático**
```bash
python setup.py
```

3. **Instale dependências**
```bash
pip install -r requirements.txt
```

4. **Verifique o sistema**
```bash
python check_system.py
```

5. **Execute a aplicação**
```bash
streamlit run streamlit_app.py
```

## 👤 Login de Teste

**Administrador:**
- Usuário: `admin`
- Senha: `admin123`

**Usuário:**
- Usuário: `usuario`  
- Senha: `user123`

## 📊 Estrutura do Projeto

```
├── streamlit_app.py          # Arquivo principal
├── modelo_preditivo.py       # Modelo de ML (backend)
├── setup.py                  # Script de inicialização
├── check_system.py           # Verificador do sistema
├── requirements.txt          # Dependências
├── auth/
│   └── auth_system.py        # Sistema de autenticação
├── data/
│   ├── store_manager.py      # Gerenciador de dados
│   ├── datasets/             # Arquivos CSV
│   └── stores_config.json    # Configuração das lojas
├── pages/
│   ├── admin.py              # Painel administrativo
│   ├── clima_vendas.py       # Análise clima x vendas
│   ├── dashboard_preditivo.py # Widgets preditivos
│   ├── modelo_preditivo.py   # Interface do modelo
│   ├── previsao_climatica.py # Previsão climática
│   └── serie_temporal.py     # Análise temporal
└── .streamlit/
    └── config.toml           # Configuração do Streamlit
```

## 🔬 Tecnologias

- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Backend**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, Statsmodels
- **Análise**: SciPy, Seaborn, Matplotlib
- **Dados**: CSV, JSON

## 📈 Modelo Preditivo

O sistema usa **Random Forest com Bootstrap Resampling** para:
- Prever vendas baseado em dados climáticos
- Gerar intervalos de confiança
- Calcular importância das features
- Validação temporal robusta

## 🌤️ Dados Climáticos

Variáveis suportadas:
- Temperatura (mín, máx, média)
- Precipitação total
- Umidade relativa
- Radiação solar
- Velocidade do vento

## 📊 Análises Disponíveis

- Correlações clima x vendas
- Decomposição sazonal
- Autocorrelação temporal
- Detecção de outliers
- Padrões por dia da semana
- Tendências de longo prazo

## 🔧 Configuração

### Adicionar Nova Loja
1. Coloque o arquivo CSV em `data/datasets/`
2. Atualize `data/stores_config.json`
3. Reinicie a aplicação

### Personalizar Modelo
- Ajuste parâmetros em "Modelo Preditivo" > "Treinamento"
- Selecione features relevantes
- Configure bootstrap e validação

## 🐛 Troubleshooting

**Erro de importação?**
```bash
python check_system.py
```

**Dados não carregam?**
- Verifique se CSV está em `data/datasets/`
- Confirme coluna 'data' no formato YYYY-MM-DD
- Valide `stores_config.json`

**Modelo não treina?**
- Mínimo 30 registros necessários
- Pelo menos 2 variáveis climáticas
- Verifique dados faltantes

## 📞 Suporte

Execute `python check_system.py` para diagnóstico automático.

---

⭐ **Versão 2.0.0** - Sistema completo com IA integrada
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ README.md criado")

def main():
    """Função principal de setup"""
    
    print("🌤️ Setup do Sistema Clima x Vendas")
    print("=" * 50)
    
    try:
        create_directory_structure()
        create_sample_data()
        create_store_config()
        create_streamlit_config()
        create_gitignore()
        create_readme()
        
        print("\n" + "=" * 50)
        print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
        print("=" * 50)
        
        print("\n📋 Próximos passos:")
        print("1. pip install -r requirements.txt")
        print("2. python check_system.py")
        print("3. streamlit run streamlit_app.py")
        
        print("\n👤 Login de teste:")
        print("   Admin: admin / admin123")
        print("   User:  usuario / user123")
        
    except Exception as e:
        print(f"\n❌ Erro durante o setup: {e}")
        print("💡 Tente executar novamente ou verificar permissões de arquivo")

if __name__ == "__main__":
    main()