# ============================================================================
# 🔍 check_system.py - VERIFICADOR DO SISTEMA
# ============================================================================

import os
import sys
import importlib
import pandas as pd
from pathlib import Path

def check_file_structure():
    """Verifica a estrutura de arquivos do projeto"""
    
    print("🔍 Verificando estrutura de arquivos...")
    
    required_files = [
        'streamlit_app.py',
        'modelo_preditivo.py',
        'requirements.txt',
        'auth/auth_system.py',
        'data/store_manager.py',
        'pages/admin.py',
        'pages/clima_vendas.py',
        'pages/dashboard_preditivo.py',
        'pages/modelo_preditivo.py',
        'pages/previsao_climatica.py',
        'pages/serie_temporal.py',
        'data/datasets/',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n📊 Resumo:")
    print(f"✅ Arquivos existentes: {len(existing_files)}")
    print(f"❌ Arquivos faltando: {len(missing_files)}")
    
    if missing_files:
        print(f"\n🚨 Arquivos faltando:")
        for file in missing_files:
            print(f"   - {file}")
    
    return len(missing_files) == 0

def check_dependencies():
    """Verifica dependências Python"""
    
    print("\n🔍 Verificando dependências Python...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'statsmodels',
        'scipy',
        'requests'
    ]
    
    missing_packages = []
    existing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            existing_packages.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    print(f"\n📊 Resumo:")
    print(f"✅ Pacotes instalados: {len(existing_packages)}")
    print(f"❌ Pacotes faltando: {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n🚨 Para instalar pacotes faltando:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def check_data_files():
    """Verifica arquivos de dados"""
    
    print("\n🔍 Verificando arquivos de dados...")
    
    data_dir = Path('data/datasets')
    
    if not data_dir.exists():
        print(f"❌ Diretório {data_dir} não existe")
        return False
    
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ Nenhum arquivo CSV encontrado em {data_dir}")
        return False
    
    print(f"✅ Encontrados {len(csv_files)} arquivo(s) CSV:")
    
    valid_files = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ {csv_file.name} - {len(df)} registros, {len(df.columns)} colunas")
            
            # Verificar colunas essenciais
            essential_cols = ['data']
            missing_cols = [col for col in essential_cols if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️ Colunas faltando: {missing_cols}")
            else:
                valid_files += 1
                
        except Exception as e:
            print(f"❌ {csv_file.name} - Erro: {e}")
    
    print(f"\n📊 Resumo:")
    print(f"✅ Arquivos válidos: {valid_files}")
    print(f"❌ Arquivos com problemas: {len(csv_files) - valid_files}")
    
    return valid_files > 0

def check_imports():
    """Verifica imports dos módulos principais"""
    
    print("\n🔍 Verificando imports dos módulos...")
    
    modules_to_test = [
        ('auth.auth_system', 'SimpleAuthenticator'),
        ('data.store_manager', 'StoreDataManager'),
        ('pages.admin', 'AdminPage'),
        ('pages.clima_vendas', 'show_clima_vendas_page'),
        ('pages.dashboard_preditivo', 'add_prediction_widgets_to_dashboard'),
        ('pages.modelo_preditivo', 'show_modelo_preditivo_page'),
        ('pages.previsao_climatica', 'show_previsao_climatica_page'),
        ('pages.serie_temporal', 'show_serie_temporal_page')
    ]
    
    successful_imports = 0
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} - ImportError: {e}")
            failed_imports.append(module_name)
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} - AttributeError: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - Error: {e}")
            failed_imports.append(module_name)
    
    print(f"\n📊 Resumo:")
    print(f"✅ Imports sucessosos: {successful_imports}")
    print(f"❌ Imports falharam: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n🚨 Módulos com problemas:")
        for module in failed_imports:
            print(f"   - {module}")
    
    return len(failed_imports) == 0

def check_streamlit_config():
    """Verifica configuração do Streamlit"""
    
    print("\n🔍 Verificando configuração do Streamlit...")
    
    config_file = Path('.streamlit/config.toml')
    
    if config_file.exists():
        print(f"✅ Arquivo de configuração encontrado: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                print(f"✅ Configuração carregada ({len(content)} caracteres)")
        except Exception as e:
            print(f"❌ Erro ao ler configuração: {e}")
            return False
    else:
        print(f"⚠️ Arquivo de configuração não encontrado: {config_file}")
        print("   Criando configuração básica...")
        
        # Criar configuração básica
        config_dir = Path('.streamlit')
        config_dir.mkdir(exist_ok=True)
        
        basic_config = """[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false
"""
        
        try:
            with open(config_file, 'w') as f:
                f.write(basic_config)
            print(f"✅ Configuração básica criada")
        except Exception as e:
            print(f"❌ Erro ao criar configuração: {e}")
            return False
    
    return True

def main():
    """Função principal de verificação"""
    
    print("🌤️ Verificador do Sistema Clima x Vendas")
    print("=" * 50)
    
    checks = [
        ("Estrutura de Arquivos", check_file_structure),
        ("Dependências Python", check_dependencies),
        ("Arquivos de Dados", check_data_files),
        ("Imports dos Módulos", check_imports),
        ("Configuração Streamlit", check_streamlit_config)
    ]
    
    results = []
    
    for check_name, check_function in checks:
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Erro na verificação {check_name}: {e}")
            results.append((check_name, False))
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO FINAL")
    print("=" * 50)
    
    passed_checks = 0
    
    for check_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {check_name}")
        if result:
            passed_checks += 1
    
    print(f"\n🎯 Resultado: {passed_checks}/{len(results)} verificações passaram")
    
    if passed_checks == len(results):
        print("🎉 SISTEMA PRONTO! Você pode executar:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ ATENÇÃO: Corrija os problemas antes de executar o sistema")
        print("\n💡 Dicas:")
        print("1. Execute: pip install -r requirements.txt")
        print("2. Verifique se todos os arquivos estão no local correto")
        print("3. Certifique-se de que há pelo menos um arquivo CSV em data/datasets/")

if __name__ == "__main__":
    main()