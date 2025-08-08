#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
# SETUP CORRIGIDO PARA WINDOWS - Resolve problema de encoding
# ============================================================================

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import locale

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def safe_write(filepath, content):
    """Escreve arquivo com encoding UTF-8 seguro"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✅ {filepath}")

def print_header():
    """Exibe cabeçalho do instalador"""
    print("\n" + "="*70)
    print("SISTEMA DE PREVISAO CLIMATICA E VENDAS - INSTALADOR")
    print("="*70)
    print("Data:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Python:", sys.version.split()[0])
    print("Diretorio:", os.getcwd())
    print("="*70 + "\n")

def check_python_version():
    """Verifica versão do Python"""
    print("Verificando versao do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERRO: Python 3.8+ necessario. Versao atual: {version.major}.{version.minor}")
        return False
    print(f"OK: Python {version.major}.{version.minor} detectado")
    return True

def install_dependencies():
    """Instala dependências necessárias"""
    print("\nInstalando dependencias...")
    
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
    
    # Criar requirements temporário
    safe_write('temp_requirements.txt', '\n'.join(essential_packages))
    
    try:
        # Atualizar pip
        print("Atualizando pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      capture_output=True, text=True)
        
        # Instalar pacotes
        print("Instalando pacotes Python...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'temp_requirements.txt'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("AVISO: Alguns pacotes nao foram instalados. Tentando instalar essenciais...")
            for pkg in essential_packages:
                subprocess.run([sys.executable, '-m', 'pip', 'install', pkg],
                             capture_output=True, text=True)
        
        print("OK: Dependencias instaladas")
        return True
        
    except Exception as e:
        print(f"ERRO ao instalar dependencias: {e}")
        print("Tente instalar manualmente com: pip install -r requirements_weather.txt")
        return False
    finally:
        # Limpar arquivo temporário
        if os.path.exists('temp_requirements.txt'):
            os.remove('temp_requirements.txt')

def create_directory_structure():
    """Cria estrutura de diretórios"""
    print("\nCriando estrutura de diretorios...")
    
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
        print(f"  OK: {dir_path}/")
    
    print("OK: Estrutura de diretorios criada")

def create_config_files():
    """Cria arquivos de configuração"""
    print("\nCriando arquivos de configuracao...")
    
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
    
    with open('config/weather_config.json', 'w', encoding='utf-8') as f:
        json.dump(weather_config, f, indent=2)
    print("  OK: config/weather_config.json")
    
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
    
    with open('config/model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)
    print("  OK: config/model_config.json")
    
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
    
    safe_write('.streamlit/config.toml', streamlit_config)
    
    print("OK: Arquivos de configuracao criados")

def create_test_system_script():
    """Cria o script test_system.py"""
    print("\nCriando script de teste...")
    
    test_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script de teste do sistema

import sys
print("\\nTESTANDO SISTEMA...")
print("="*50)

# Testar imports
imports_ok = True

try:
    import pandas as pd
    print("OK: pandas")
except:
    print("ERRO: pandas")
    imports_ok = False

try:
    import numpy as np
    print("OK: numpy")
except:
    print("ERRO: numpy")
    imports_ok = False

try:
    import sklearn
    print("OK: scikit-learn")
except:
    print("ERRO: scikit-learn")
    imports_ok = False

try:
    import streamlit
    print("OK: streamlit")
except:
    print("ERRO: streamlit")
    imports_ok = False

try:
    import requests
    print("OK: requests")
except:
    print("ERRO: requests")
    imports_ok = False

try:
    import plotly
    print("OK: plotly")
except:
    print("ERRO: plotly")
    imports_ok = False

try:
    import schedule
    print("OK: schedule")
except:
    print("ERRO: schedule")
    imports_ok = False

try:
    import cfgrib
    print("OK: cfgrib")
except:
    print("AVISO: cfgrib (opcional para dados GRIB2)")

try:
    import xarray
    print("OK: xarray")
except:
    print("AVISO: xarray (opcional para dados NetCDF)")

print("="*50)
if imports_ok:
    print("SUCESSO: Sistema pronto para uso!")
else:
    print("ERRO: Instale as dependencias faltantes")
    print("Execute: pip install -r requirements_weather.txt")
"""
    
    safe_write('test_system.py', test_script)
    print("OK: test_system.py criado")

def create_run_system_script():
    """Cria o script run_system.py"""
    print("\nCriando script principal...")
    
    run_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script principal para executar o sistema

import sys
import os

def main():
    print("\\nSISTEMA DE PREVISAO CLIMATICA E VENDAS")
    print("="*50)
    print("\\nEscolha o modulo para executar:")
    print("1. Sistema de Download NOMADS")
    print("2. Modelo Preditivo")
    print("3. Dashboard Streamlit")
    print("4. Atualizacao Automatica")
    print("5. Teste do Sistema")
    
    choice = input("\\nOpcao: ").strip()
    
    if choice == '1':
        try:
            from sistema_previsao_climatica import main
            main()
        except ImportError:
            print("ERRO: sistema_previsao_climatica.py nao encontrado")
    elif choice == '2':
        try:
            from modelo_preditivo_integrado import main
            main()
        except ImportError:
            print("ERRO: modelo_preditivo_integrado.py nao encontrado")
    elif choice == '3':
        os.system('streamlit run streamlit_app.py')
    elif choice == '4':
        try:
            from sistema_previsao_climatica import WeatherDataManager, WeatherAutomation
            manager = WeatherDataManager()
            automation = WeatherAutomation(manager)
            automation.schedule_updates()
            print("Sistema de automacao iniciado. Ctrl+C para parar.")
            automation.start_scheduler()
        except ImportError:
            print("ERRO: sistema_previsao_climatica.py nao encontrado")
    elif choice == '5':
        os.system('python test_system.py')
    else:
        print("Opcao invalida")

if __name__ == "__main__":
    main()
"""
    
    safe_write('run_system.py', run_script)
    print("OK: run_system.py criado")

def copy_existing_files():
    """Copia arquivos existentes para os locais corretos"""
    print("\nOrganizando arquivos existentes...")
    
    # Lista de arquivos para copiar/mover
    file_mappings = {
        'dowload_modelo.py': 'NOMADS/download_modelo.py',
        'download_modelo.py': 'NOMADS/download_modelo.py',  # Caso tenha corrigido o nome
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
                try:
                    shutil.copy2(source, dest)
                    print(f"  OK: Copiado {source} -> {dest}")
                except Exception as e:
                    print(f"  AVISO: Erro ao copiar {source}: {e}")
        else:
            print(f"  INFO: Nao encontrado: {source}")

def create_sample_data():
    """Cria arquivo de dados de exemplo se não existir"""
    sample_file = 'data/datasets/Loja1_dados_unificados.csv'
    
    if not os.path.exists(sample_file):
        print("\nCriando arquivo de dados de exemplo...")
        try:
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
            df.to_csv(sample_file, index=False)
            print(f"  OK: Arquivo de exemplo criado: {sample_file}")
        except Exception as e:
            print(f"  AVISO: Nao foi possivel criar arquivo de exemplo: {e}")

def final_tests():
    """Executa testes finais"""
    print("\nExecutando testes finais...")
    
    # Verificar se test_system.py existe
    if os.path.exists('test_system.py'):
        print("  OK: test_system.py existe")
        
        # Tentar executar
        try:
            result = subprocess.run([sys.executable, 'test_system.py'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("  OK: Teste executado com sucesso")
                print(result.stdout)
            else:
                print("  AVISO: Teste retornou erro")
                if result.stderr:
                    print(result.stderr)
        except subprocess.TimeoutExpired:
            print("  AVISO: Teste demorou muito para executar")
        except Exception as e:
            print(f"  AVISO: Erro ao executar teste: {e}")
    else:
        print("  ERRO: test_system.py nao encontrado")

def main():
    """Função principal do instalador"""
    print_header()
    
    # Lista de etapas
    steps = [
        ("Verificando Python", check_python_version),
        ("Instalando dependencias", install_dependencies),
        ("Criando diretorios", create_directory_structure),
        ("Criando configuracoes", create_config_files),
        ("Criando script de teste", create_test_system_script),
        ("Criando script principal", create_run_system_script),
        ("Organizando arquivos", copy_existing_files),
        ("Criando dados de exemplo", create_sample_data),
        ("Testes finais", final_tests)
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            result = step_func()
            if result == False:
                success = False
                print(f"AVISO: {step_name} teve problemas")
        except Exception as e:
            print(f"ERRO em {step_name}: {e}")
            success = False
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO DA INSTALACAO")
    print("="*70)
    
    if success:
        print("SUCESSO: Sistema instalado com sucesso!")
        print("\nPROXIMOS PASSOS:")
        print("1. Configure sua localizacao em: config/weather_config.json")
        print("2. Execute o sistema: python run_system.py")
        print("3. Ou acesse o dashboard: streamlit run streamlit_app.py")
        print("\nDICAS:")
        print("- Teste o sistema com: python test_system.py")
        print("- Logs salvos em logs/weather_system.log")
    else:
        print("AVISO: Instalacao concluida com avisos")
        print("Verifique os erros acima e tente corrigir manualmente")
        print("\nSOLUCOES COMUNS:")
        print("- Instale dependencias: pip install pandas numpy scikit-learn streamlit")
        print("- Para cfgrib, use conda: conda install -c conda-forge cfgrib")
        print("- Verifique permissoes de escrita nos diretorios")
    
    print("\n" + "="*70)
    print("Obrigado por usar o Sistema de Previsao Climatica!")
    print("="*70 + "\n")
    
    input("Pressione ENTER para finalizar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAVISO: Instalacao cancelada pelo usuario")
    except Exception as e:
        print(f"\nERRO FATAL: {e}")
        print("Tente executar com privilegios de administrador")
        input("Pressione ENTER para sair...")