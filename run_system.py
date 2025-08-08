#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script principal para executar o sistema

import sys
import os

def main():
    print("\nSISTEMA DE PREVISAO CLIMATICA E VENDAS")
    print("="*50)
    print("\nEscolha o modulo para executar:")
    print("1. Sistema de Download NOMADS")
    print("2. Modelo Preditivo")
    print("3. Dashboard Streamlit")
    print("4. Atualizacao Automatica")
    print("5. Teste do Sistema")
    
    choice = input("\nOpcao: ").strip()
    
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
