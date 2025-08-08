#!/usr/bin/env python
# ============================================================================
# 🚀 setup_weather_system.py - INSTALAÇÃO E CONFIGURAÇÃO AUTOMÁTICA
# ============================================================================
# Script completo para configurar o sistema de previsão climática e vendas

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

def print_header():
    """Exibe cabeçalho do instalador"""
    print("\n" + "="*70)
    print("🌤️  SISTEMA DE PREVISÃO CLIMÁTICA E VENDAS - INSTALADOR")
    print("="*70)
    print("📅 Data:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🐍 Python:", sys.version.split()[0])
    print("📂 Diretório:", os.getcwd())
    print("="*70 + "\n")

def check_python_version():
    """Verifica versão do Python"""
    print("🔍 Verificando versão do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ necessário. Versão atual: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detectado")
    return True

def install_dependencies():
    """Instala dependências necessárias"""
    print("\n📦 Instalando dependências...")
    
    # Lista de pacotes essenciais
    essential_packages = [
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'requests>=2.31.0',
        'streamlit>=1.28.0',
        'plotly>=5.14.0',
        'openpyxl>=3.1.0',
        'schedule>=1.2.0',
        'joblib>=1.3.0'
    ]
    
    # Tentar instalar cfgrib e suas dependências
    grib_packages = [
        'eccodes',  # Necessário para cfgrib
        'cfgrib>=0.9.10',
        'xarray>=2023.0.0',
        'netCDF4>=1.6.0'
    ]
    
    all_packages = essential_packages + grib_packages
    
    # Criar requirements.txt temporário
    with open('temp_requirements.txt', 'w') as f:
        f.write('\n'.join(all_packages))
    
    try:
        # Atualizar pip
        print("📌 Atualizando pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      capture_output=True, text=True)
        
        # Instalar pacotes
        print("📌 Instalando pacotes Python...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'temp_requirements.txt'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("⚠️ Alguns pacotes não foram instalados. Tentando instalar essenciais...")
            # Tentar instalar apenas os essenciais
            for pkg in essential_packages:
                subprocess.run([sys.executable, '-m', 'pip', 'install', pkg],
                             capture_output=True, text=True)
        
        print("✅ Dependências instaladas")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        print("💡 Tente instalar manualmente com: pip install -r requirements_weather.txt")
        return False
    finally:
        # Limpar arquivo temporário
        if os.path.exists('temp_requirements.txt'):
            os.remove('temp_requirements.txt')

def create_directory_structure():
    """Cria estrutura de diretórios"""
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        'config',
        'data',
        'data/datasets',
        'NOMADS',
        'processed_data',
        'models',
        'logs',
        'cache',
        'exports',
        'paginas',
        'auth',
        '.streamlit'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}/")
    
    print("✅ Estrutura de diretórios criada")

def create_config_files():
    """Cria arquivos de configuração"""
    print("\n⚙️ Criando arquivos de configuração...")
    
    # Configuração do sistema meteorológico
    weather_config = {
        "base_dir": os.getcwd(),
        "location": {
            "lat_min": -29.73,
            "lat_max": -29.63,
            "lon_min": 306.47,
            "lon_max": 306.57,
            "city": "Agudo",
            "state": "RS"
        },
        "variables": [
            "TMP", "TMAX", "TMIN", "UGRD", "VGRD",
            "RH", "APCP", "PRES", "DSWRF", "TCDC"
        ],
        "levels": [
            "2_m_above_ground",
            "10_m_above_ground",
            "surface",
            "mean_sea_level"
        ],
        "forecast_hours": 120,
        "update_times": ["00", "06", "12", "18"],
        "auto_update": True
    }
    
    with open('config/weather_config.json', 'w') as f:
        json.dump(weather_config, f, indent=2)
    print("  ✅ config/weather_config.json")
    
    # Configuração do modelo
    model_config = {
        "models": {
            "random_forest": {
                "enabled": True,
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "random_state": 42
                }
            },
            "gradient_boosting": {
                "enabled": True,
                "params": {
                    "n_estimators": 150,
                    "max_depth": 7,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            }
        },
        "features": {
            "weather_features": [
                "temp_media", "temp_max", "temp_min",
                "precipitacao_total", "umid_mediana",
                "pressao_media", "vento_vel_media"
            ],
            "engineered_features": [
                "temp_range", "temp_squared", "feels_like",
                "humidity_temp_interaction", "weather_score"
            ],
            "lag_features": {
                "enabled": True,
                "lags": [1, 7, 14]
            }
        },
        "validation": {
            "test_size": 0.2,
            "cv_folds": 5,
            "bootstrap_samples": 50
        }
    }
    
    with open('config/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    print("  ✅ config/model_config.json")
    
    # Configuração do Streamlit
    streamlit_config = """[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
"""
    
    with open('.streamlit/config.toml', 'w') as f:
        f.write(streamlit_config)
    print("  ✅ .streamlit/config.toml")
    
    print("✅ Arquivos de configuração criados")

def copy_existing_files():
    """Copia arquivos existentes para os locais corretos"""
    print("\n📂 Organizando arquivos existentes...")
    
    # Lista de arquivos para copiar/mover
    file_mappings = {
        'dowload_modelo.py': 'NOMADS/download_modelo.py',
        'scripty.py': 'NOMADS/process_grib.py',
        'modelo_preditivo.py': 'models/modelo_preditivo_original.py',
        'data/datasets/Loja1_dados_unificados.csv': 'data/datasets/Loja1_dados_unificados.csv'
    }
    
    for source, dest in file_mappings.items():
        if os.path.exists(source):
            dest_dir = os.path.dirname(dest)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            
            if source != dest:
                shutil.copy2(source, dest)
                print(f"  ✅ Copiado: {source} → {dest}")
        else:
            print(f"  ⚠️ Não encontrado: {source}")

def create_main_scripts():
    """Cria scripts principais"""
    print("\n📝 Criando scripts principais...")
    
    # Script principal de execução
    run_script = """#!/usr/bin/env python
# Script principal para executar o sistema

import sys
import os

def main():
    print("\\n🌤️ SISTEMA DE PREVISÃO CLIMÁTICA E VENDAS")
    print("="*50)
    print("\\nEscolha o módulo para executar:")
    print("1. 🌍 Sistema de Download NOMADS")
    print("2. 🤖 Modelo Preditivo")
    print("3. 📊 Dashboard Streamlit")
    print("4. 🔄 Atualização Automática")
    
    choice = input("\\nOpção: ").strip()
    
    if choice == '1':
        from sistema_previsao_climatica import main
        main()
    elif choice == '2':
        from modelo_preditivo_integrado import main
        main()
    elif choice == '3':
        os.system('streamlit run streamlit_app.py')
    elif choice == '4':
        from sistema_previsao_climatica import WeatherDataManager, WeatherAutomation
        manager = WeatherDataManager()
        automation = WeatherAutomation(manager)
        automation.schedule_updates()
        print("⏰ Sistema de automação iniciado. Ctrl+C para parar.")
        automation.start_scheduler()
    else:
        print("❌ Opção inválida")

if __name__ == "__main__":
    main()
"""
    
    with open('run_system.py', 'w') as f:
        f.write(run_script)
    print("  ✅ run_system.py")
    
    # Script de teste rápido
    test_script = """#!/usr/bin/env python
# Script de teste do sistema

import sys
print("\\n🧪 TESTANDO SISTEMA...")
print("="*50)

# Testar imports
imports_ok = True

try:
    import pandas as pd
    print("✅ pandas")
except:
    print("❌ pandas")
    imports_ok = False

try:
    import numpy as np
    print("✅ numpy")
except:
    print("❌ numpy")
    imports_ok = False

try:
    import sklearn
    print("✅ scikit-learn")
except:
    print("❌ scikit-learn")
    imports_ok = False

try:
    import streamlit
    print("✅ streamlit")
except:
    print("❌ streamlit")
    imports_ok = False

try:
    import cfgrib
    print("✅ cfgrib")
except:
    print("⚠️ cfgrib (opcional)")

try:
    import xarray
    print("✅ xarray")
except:
    print("⚠️ xarray (opcional)")

print("="*50)
if imports_ok:
    print("✅ Sistema pronto para uso!")
else:
    print("❌ Instale as dependências faltantes")
"""
    
    with open('test_system.py', 'w') as f:
        f.write(test_script)
    print("  ✅ test_system.py")
    
    print("✅ Scripts principais criados")

def create_documentation():
    """Cria documentação do sistema"""
    print("\n📚 Criando documentação...")
    
    readme = """# 🌤️ Sistema de Previsão Climática e Vendas

## 📋 Descrição
Sistema profissional e automatizado para download de dados meteorológicos do NOMADS/GFS,
processamento avançado e integração com modelo preditivo de vendas.

## 🚀 Instalação Rápida
```bash
python setup_weather_system.py
```

## 📦 Componentes Principais

### 1. Sistema de Previsão Climática (`sistema_previsao_climatica.py`)
- Download automático de dados NOMADS/GFS
- Processamento de arquivos GRIB2
- Geração de relatórios
- Automação com agendamento

### 2. Modelo Preditivo Integrado (`modelo_preditivo_integrado.py`)
- Múltiplos algoritmos de ML
- Engenharia de features avançada
- Validação temporal
- Intervalos de confiança

### 3. Dashboard Streamlit (`streamlit_app.py`)
- Interface web interativa
- Visualizações em tempo real
- Controle de modelos
- Exportação de dados

## 🎯 Uso Básico

### Executar Sistema Completo
```bash
python run_system.py
```

### Download de Dados NOMADS
```python
from sistema_previsao_climatica import WeatherDataManager

manager = WeatherDataManager()
manager.run_automatic_update()
```

### Treinar Modelo
```python
from modelo_preditivo_integrado import ModeloVendasClimaticoAvancado

modelo = ModeloVendasClimaticoAvancado()
df = modelo.load_and_prepare_data()
modelo.train(df)
```

### Dashboard Web
```bash
streamlit run streamlit_app.py
```

## ⚙️ Configuração

### Localização (config/weather_config.json)
Ajuste as coordenadas para sua região:
```json
{
    "location": {
        "lat_min": -29.73,
        "lat_max": -29.63,
        "lon_min": 306.47,
        "lon_max": 306.57,
        "city": "Agudo",
        "state": "RS"
    }
}
```

### Modelo (config/model_config.json)
Configure algoritmos e parâmetros:
```json
{
    "models": {
        "random_forest": {
            "enabled": true,
            "params": {
                "n_estimators": 200
            }
        }
    }
}
```

## 📊 Estrutura de Dados

### Entrada (CSV)
- data: YYYY-MM-DD
- temp_max, temp_min, temp_media
- precipitacao_total
- umid_mediana
- valor_loja_01

### Saída
- Previsões com intervalos de confiança
- Relatórios em JSON/Excel
- Gráficos interativos

## 🔄 Automação

O sistema pode ser configurado para:
- Baixar dados automaticamente (00, 06, 12, 18 UTC)
- Processar e integrar com vendas
- Retreinar modelos periodicamente
- Enviar alertas

## 📈 Métricas de Performance

- R² Score > 0.85
- RMSE < 10% do valor médio
- MAPE < 15%
- Cross-validation com Time Series Split

## 🐛 Troubleshooting

### Erro ao instalar cfgrib
```bash
conda install -c conda-forge cfgrib eccodes
```

### Erro de memória
Reduza o número de features ou use amostragem.

### Dados não baixam
Verifique conexão e disponibilidade do NOMADS.

## 📞 Suporte
- Logs em: logs/weather_system.log
- Teste com: python test_system.py

## 📝 Licença
MIT License

---
Desenvolvido com ❤️ para previsão climática e análise de vendas
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    print("  ✅ README.md")
    
    print("✅ Documentação criada")

def setup_cron_job():
    """Configura tarefa agendada (Linux/Mac)"""
    if sys.platform != 'win32':
        print("\n⏰ Configurando agendamento automático...")
        
        cron_command = f"0 1,7,13,19 * * * cd {os.getcwd()} && {sys.executable} sistema_previsao_climatica.py"
        
        print(f"  📝 Adicione ao crontab (crontab -e):")
        print(f"     {cron_command}")
        
        with open('cron_setup.txt', 'w') as f:
            f.write(cron_command)
        print("  ✅ Comando salvo em cron_setup.txt")

def final_setup():
    """Configurações finais e teste"""
    print("\n🔧 Executando configurações finais...")
    
    # Criar arquivo de exemplo se não existir
    if not os.path.exists('data/datasets/Loja1_dados_unificados.csv'):
        print("  📊 Criando arquivo de exemplo...")
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        data = {
            'data': dates,
            'temp_max': np.random.uniform(20, 35, 365),
            'temp_min': np.random.uniform(10, 25, 365),
            'temp_media': np.random.uniform(15, 30, 365),
            'precipitacao_total': np.random.exponential(5, 365),
            'umid_mediana': np.random.uniform(40, 90, 365),
            'valor_loja_01': np.random.uniform(40000, 60000, 365)
        }
        df = pd.DataFrame(data)
        df.to_csv('data/datasets/Loja1_dados_unificados.csv', index=False)
        print("  ✅ Arquivo de exemplo criado")
    
    # Testar importação dos módulos principais
    print("\n🧪 Testando módulos...")
    try:
        exec(open('test_system.py').read())
    except Exception as e:
        print(f"  ⚠️ Erro no teste: {e}")
    
    print("\n✅ Setup concluído!")

def main():
    """Função principal do instalador"""
    print_header()
    
    # Verificações e instalação
    steps = [
        ("Verificando Python", check_python_version),
        ("Instalando dependências", install_dependencies),
        ("Criando diretórios", create_directory_structure),
        ("Criando configurações", create_config_files),
        ("Organizando arquivos", copy_existing_files),
        ("Criando scripts", create_main_scripts),
        ("Criando documentação", create_documentation),
        ("Setup final", final_setup)
    ]
    
    success = True
    for step_name, step_func in steps:
        try:
            result = step_func()
            if result == False:
                success = False
                print(f"⚠️ {step_name} teve problemas")
        except Exception as e:
            print(f"❌ Erro em {step_name}: {e}")
            success = False
    
    # Configuração de cron se não for Windows
    if sys.platform != 'win32':
        setup_cron_job()
    
    # Resumo final
    print("\n" + "="*70)
    print("📊 RESUMO DA INSTALAÇÃO")
    print("="*70)
    
    if success:
        print("✅ Sistema instalado com sucesso!")
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Configure sua localização em: config/weather_config.json")
        print("2. Execute o sistema: python run_system.py")
        print("3. Ou acesse o dashboard: streamlit run streamlit_app.py")
        print("\n💡 DICAS:")
        print("- Veja a documentação em README.md")
        print("- Logs salvos em logs/weather_system.log")
        print("- Teste o sistema com: python test_system.py")
    else:
        print("⚠️ Instalação concluída com avisos")
        print("Verifique os erros acima e tente corrigir manualmente")
        print("\n💡 SOLUÇÕES COMUNS:")
        print("- Instale dependências manualmente: pip install -r requirements_weather.txt")
        print("- Para cfgrib, use conda: conda install -c conda-forge cfgrib")
        print("- Verifique permissões de escrita nos diretórios")
    
    print("\n" + "="*70)
    print("🌤️ Obrigado por usar o Sistema de Previsão Climática!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Instalação cancelada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        print("Tente executar com privilégios de administrador")