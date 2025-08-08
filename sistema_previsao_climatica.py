# ============================================================================
# 🌤️ sistema_previsao_climatica.py - SISTEMA COMPLETO DE PREVISÃO CLIMÁTICA
# ============================================================================
# Sistema profissional para download, processamento e análise de dados NOMADS/GFS
# com integração ao modelo preditivo de vendas

import os
import sys
import requests
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import schedule
import time
import logging
from pathlib import Path
import subprocess

warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherDataManager:
    """Gerenciador completo de dados meteorológicos NOMADS/GFS"""
    
    def __init__(self, config_file='config/weather_config.json'):
        """Inicializa o gerenciador com configurações"""
        self.config = self.load_config(config_file)
        self.base_dir = Path(self.config.get('base_dir', os.getcwd()))
        self.data_dir = self.base_dir / "NOMADS"
        self.processed_dir = self.base_dir / "processed_data"
        self.create_directories()
        
    def load_config(self, config_file):
        """Carrega configurações do sistema"""
        default_config = {
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
            "auto_update": True,
            "notification_email": None
        }
        
        # Tentar carregar configuração personalizada
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
                logger.info(f"✅ Configuração carregada de {config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar config: {e}. Usando padrão.")
        else:
            # Criar arquivo de configuração se não existir
            os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"📝 Arquivo de configuração criado: {config_file}")
        
        return default_config
    
    def create_directories(self):
        """Cria estrutura de diretórios necessária"""
        dirs = [
            self.data_dir,
            self.processed_dir,
            self.base_dir / "logs",
            self.base_dir / "cache",
            self.base_dir / "exports"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Estrutura de diretórios criada")
    
    def get_latest_gfs_cycle(self):
        """Obtém o ciclo GFS mais recente disponível"""
        now = datetime.utcnow()
        cycles = ['00', '06', '12', '18']
        
        # Testar ciclos do dia atual
        for i in range(4):
            test_time = now - timedelta(hours=i*6)
            cycle_hour = cycles[int(test_time.hour/6)]
            cycle_date = test_time.strftime('%Y%m%d')
            
            # Verificar se o ciclo está disponível
            test_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{cycle_date}%2F{cycle_hour}%2Fatmos"
            
            try:
                response = requests.head(test_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Ciclo GFS encontrado: {cycle_date}/{cycle_hour}Z")
                    return cycle_date, cycle_hour
            except:
                continue
        
        # Se não encontrar, usar o mais provável
        cycle_date = (now - timedelta(hours=6)).strftime('%Y%m%d')
        cycle_hour = cycles[int((now.hour - 6)/6)]
        logger.warning(f"⚠️ Usando ciclo padrão: {cycle_date}/{cycle_hour}Z")
        return cycle_date, cycle_hour
    
    def download_gfs_data(self, force_update=False):
        """Baixa dados GFS do NOMADS"""
        cycle_date, cycle_hour = self.get_latest_gfs_cycle()
        
        logger.info(f"📥 Iniciando download GFS {cycle_date}/{cycle_hour}Z")
        
        base_url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
        
        # Construir parâmetros da requisição
        location = self.config['location']
        params = []
        
        # Adicionar variáveis
        for var in self.config['variables']:
            params.append(f'&var_{var}=on')
        
        # Adicionar níveis
        for level in self.config['levels']:
            params.append(f'&lev_{level}=on')
        
        # Adicionar região
        params.append(f'&subregion=')
        params.append(f'&toplat={location["lat_max"]}')
        params.append(f'&leftlon={location["lon_min"]}')
        params.append(f'&rightlon={location["lon_max"]}')
        params.append(f'&bottomlat={location["lat_min"]}')
        
        params_str = ''.join(params)
        
        # Criar diretório para este ciclo
        cycle_dir = self.data_dir / f"{cycle_date}_{cycle_hour}Z"
        cycle_dir.mkdir(exist_ok=True)
        
        downloaded_files = []
        errors = []
        
        # Baixar arquivos para cada hora de previsão
        for fhour in range(0, self.config['forecast_hours'] + 1, 3):
            fstr = f'{fhour:03d}'
            filename = f'gfs.t{cycle_hour}z.pgrb2.0p25.f{fstr}'
            filepath = cycle_dir / filename
            
            # Pular se já existe e não forçar atualização
            if filepath.exists() and not force_update:
                logger.info(f"✅ Já existe: {filename}")
                downloaded_files.append(filepath)
                continue
            
            # Construir URL completa
            dir_param = f'%2Fgfs.{cycle_date}%2F{cycle_hour}%2Fatmos'
            url = f'{base_url}?file={filename}{params_str}&dir={dir_param}'
            
            # Tentar baixar o arquivo
            try:
                logger.info(f"⬇️ Baixando: {filename}")
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Salvar arquivo
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verificar tamanho
                    file_size = filepath.stat().st_size
                    if file_size > 1000:  # Arquivo válido deve ter mais de 1KB
                        logger.info(f"✅ Salvo: {filename} ({file_size/1024/1024:.2f} MB)")
                        downloaded_files.append(filepath)
                    else:
                        filepath.unlink()
                        errors.append(f"Arquivo vazio: {filename}")
                        logger.error(f"❌ Arquivo vazio: {filename}")
                else:
                    errors.append(f"HTTP {response.status_code}: {filename}")
                    logger.error(f"❌ Erro HTTP {response.status_code}: {filename}")
                    
            except Exception as e:
                errors.append(f"Erro em {filename}: {str(e)}")
                logger.error(f"❌ Erro ao baixar {filename}: {e}")
        
        # Resumo do download
        logger.info(f"📊 Download concluído: {len(downloaded_files)} arquivos baixados, {len(errors)} erros")
        
        return {
            'success': len(downloaded_files) > 0,
            'files': downloaded_files,
            'errors': errors,
            'cycle': f"{cycle_date}_{cycle_hour}Z"
        }
    
    def process_grib_files(self, files):
        """Processa arquivos GRIB2 e extrai dados meteorológicos"""
        logger.info(f"🔧 Processando {len(files)} arquivos GRIB2...")
        
        all_data = []
        
        for file_path in files:
            try:
                # Abrir arquivo GRIB2
                ds = xr.open_dataset(
                    file_path, 
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""}
                )
                
                # Extrair hora de previsão do nome do arquivo
                filename = file_path.name
                fhour = int(filename.split('.f')[1][:3])
                
                # Calcular data/hora válida
                base_time = pd.to_datetime(ds.time.values)
                valid_time = base_time + pd.Timedelta(hours=fhour)
                
                # Extrair variáveis disponíveis
                data_point = {
                    'data': valid_time.date(),
                    'hora': valid_time.time(),
                    'forecast_hour': fhour
                }
                
                # Mapear variáveis GFS para nomes padronizados
                variable_mapping = {
                    't2m': 'temp_2m',
                    'tmax': 'temp_max',
                    'tmin': 'temp_min',
                    'r2': 'umidade_2m',
                    'tp': 'precipitacao',
                    'sp': 'pressao_superficie',
                    'msl': 'pressao_mar',
                    'u10': 'vento_u_10m',
                    'v10': 'vento_v_10m',
                    'dswrf': 'radiacao_solar',
                    'tcc': 'cobertura_nuvens'
                }
                
                # Extrair valores médios para a região
                for gfs_var, nossa_var in variable_mapping.items():
                    if gfs_var in ds:
                        try:
                            value = float(ds[gfs_var].mean().values)
                            
                            # Conversões de unidade
                            if 'temp' in nossa_var and gfs_var != 'tp':
                                value = value - 273.15  # Kelvin para Celsius
                            elif nossa_var == 'pressao_superficie':
                                value = value / 100  # Pa para hPa
                            elif nossa_var == 'pressao_mar':
                                value = value / 100  # Pa para hPa
                            
                            data_point[nossa_var] = round(value, 2)
                        except:
                            continue
                
                # Calcular velocidade e direção do vento
                if 'vento_u_10m' in data_point and 'vento_v_10m' in data_point:
                    u = data_point['vento_u_10m']
                    v = data_point['vento_v_10m']
                    data_point['vento_velocidade'] = round(np.sqrt(u**2 + v**2), 2)
                    data_point['vento_direcao'] = round(np.degrees(np.arctan2(v, u)) % 360, 0)
                
                all_data.append(data_point)
                
                # Fechar dataset
                ds.close()
                
                logger.info(f"✅ Processado: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {file_path.name}: {e}")
                continue
        
        # Criar DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values(['data', 'hora', 'forecast_hour'])
            logger.info(f"✅ Dados processados: {len(df)} registros")
            return df
        else:
            logger.error("❌ Nenhum dado foi processado")
            return pd.DataFrame()
    
    def save_processed_data(self, df, format='all'):
        """Salva dados processados em diferentes formatos"""
        if df.empty:
            logger.warning("⚠️ DataFrame vazio, nada para salvar")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = []
        
        # Salvar CSV
        if format in ['csv', 'all']:
            csv_path = self.processed_dir / f"weather_data_{timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            saved_files.append(csv_path)
            logger.info(f"💾 CSV salvo: {csv_path}")
        
        # Salvar Excel
        if format in ['excel', 'all']:
            excel_path = self.processed_dir / f"weather_data_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados Meteorológicos', index=False)
                
                # Adicionar metadados
                metadata = pd.DataFrame({
                    'Informação': ['Data de Processamento', 'Total de Registros', 
                                   'Período', 'Localização'],
                    'Valor': [
                        timestamp,
                        len(df),
                        f"{df['data'].min()} a {df['data'].max()}",
                        f"{self.config['location']['city']}, {self.config['location']['state']}"
                    ]
                })
                metadata.to_excel(writer, sheet_name='Metadados', index=False)
            
            saved_files.append(excel_path)
            logger.info(f"💾 Excel salvo: {excel_path}")
        
        # Salvar JSON
        if format in ['json', 'all']:
            json_path = self.processed_dir / f"weather_data_{timestamp}.json"
            df.to_json(json_path, orient='records', date_format='iso', indent=2)
            saved_files.append(json_path)
            logger.info(f"💾 JSON salvo: {json_path}")
        
        # Salvar Parquet (eficiente para big data)
        if format in ['parquet', 'all']:
            parquet_path = self.processed_dir / f"weather_data_{timestamp}.parquet"
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            saved_files.append(parquet_path)
            logger.info(f"💾 Parquet salvo: {parquet_path}")
        
        # Atualizar link simbólico para dados mais recentes
        latest_link = self.processed_dir / "latest_weather_data.csv"
        if latest_link.exists():
            latest_link.unlink()
        if saved_files and format in ['csv', 'all']:
            os.symlink(saved_files[0], latest_link)
            logger.info(f"🔗 Link simbólico atualizado: {latest_link}")
        
        return saved_files
    
    def merge_with_sales_data(self, weather_df, sales_csv_path):
        """Mescla dados meteorológicos com dados de vendas"""
        try:
            # Carregar dados de vendas
            sales_df = pd.read_csv(sales_csv_path)
            
            # Garantir que a coluna de data está no formato correto
            sales_df['data'] = pd.to_datetime(sales_df['data'])
            weather_df['data'] = pd.to_datetime(weather_df['data'])
            
            # Agregar dados meteorológicos por dia (médias)
            weather_daily = weather_df.groupby('data').agg({
                'temp_2m': 'mean',
                'temp_max': 'max',
                'temp_min': 'min',
                'umidade_2m': 'mean',
                'precipitacao': 'sum',
                'pressao_superficie': 'mean',
                'vento_velocidade': 'mean',
                'radiacao_solar': 'mean',
                'cobertura_nuvens': 'mean'
            }).round(2)
            
            # Renomear colunas para padrão do projeto
            weather_daily.columns = [
                'temp_media', 'temp_max', 'temp_min', 'umid_mediana',
                'precipitacao_total', 'pressao_media', 'vento_vel_media',
                'rad_mediana', 'nuvens_media'
            ]
            
            # Fazer merge com dados de vendas
            merged_df = pd.merge(
                sales_df,
                weather_daily,
                on='data',
                how='outer',
                suffixes=('_vendas', '_clima')
            )
            
            # Ordenar por data
            merged_df = merged_df.sort_values('data')
            
            # Salvar resultado
            output_path = self.processed_dir / f"dados_unificados_{datetime.now().strftime('%Y%m%d')}.csv"
            merged_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Dados unificados salvos: {output_path}")
            logger.info(f"📊 Total de registros: {len(merged_df)}")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"❌ Erro ao mesclar dados: {e}")
            return None
    
    def run_automatic_update(self):
        """Executa atualização automática completa"""
        logger.info("🔄 Iniciando atualização automática...")
        
        try:
            # 1. Baixar dados mais recentes
            download_result = self.download_gfs_data()
            
            if not download_result['success']:
                logger.error("❌ Falha no download dos dados")
                return False
            
            # 2. Processar arquivos GRIB
            df = self.process_grib_files(download_result['files'])
            
            if df.empty:
                logger.error("❌ Nenhum dado processado")
                return False
            
            # 3. Salvar dados processados
            saved_files = self.save_processed_data(df)
            
            # 4. Mesclar com dados de vendas se existir
            sales_file = self.base_dir / "data" / "datasets" / "Loja1_dados_unificados.csv"
            if sales_file.exists():
                merged_df = self.merge_with_sales_data(df, sales_file)
                
                # Atualizar arquivo de vendas com dados climáticos mais recentes
                if merged_df is not None:
                    merged_df.to_csv(sales_file, index=False)
                    logger.info(f"✅ Arquivo de vendas atualizado com dados climáticos")
            
            # 5. Gerar relatório
            self.generate_report(df)
            
            logger.info("✅ Atualização automática concluída com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na atualização automática: {e}")
            return False
    
    def generate_report(self, df):
        """Gera relatório resumido dos dados"""
        if df.empty:
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'date_range': {
                'start': str(df['data'].min()),
                'end': str(df['data'].max())
            },
            'temperature': {
                'min': float(df['temp_2m'].min()) if 'temp_2m' in df else None,
                'max': float(df['temp_2m'].max()) if 'temp_2m' in df else None,
                'mean': float(df['temp_2m'].mean()) if 'temp_2m' in df else None
            },
            'precipitation': {
                'total': float(df['precipitacao'].sum()) if 'precipitacao' in df else None,
                'days_with_rain': int((df['precipitacao'] > 0).sum()) if 'precipitacao' in df else None
            },
            'location': self.config['location']
        }
        
        # Salvar relatório
        report_path = self.processed_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Relatório gerado: {report_path}")
        
        # Imprimir resumo
        print("\n" + "="*50)
        print("📊 RESUMO DOS DADOS METEOROLÓGICOS")
        print("="*50)
        print(f"📅 Período: {report['date_range']['start']} a {report['date_range']['end']}")
        print(f"📈 Total de registros: {report['total_records']}")
        if report['temperature']['mean']:
            print(f"🌡️ Temperatura média: {report['temperature']['mean']:.1f}°C")
        if report['precipitation']['total']:
            print(f"🌧️ Precipitação total: {report['precipitation']['total']:.1f}mm")
        print("="*50 + "\n")
        
        return report

class WeatherAutomation:
    """Sistema de automação para coleta e processamento de dados"""
    
    def __init__(self, weather_manager):
        self.weather_manager = weather_manager
        self.scheduler_active = False
    
    def schedule_updates(self):
        """Configura agendamento de atualizações automáticas"""
        # Agendar para cada ciclo GFS (00, 06, 12, 18 UTC)
        update_times = self.weather_manager.config['update_times']
        
        for hour in update_times:
            # Adicionar 30 minutos de delay para garantir disponibilidade
            schedule_time = f"{int(hour):02d}:30"
            schedule.every().day.at(schedule_time).do(self.weather_manager.run_automatic_update)
            logger.info(f"⏰ Atualização agendada para {schedule_time} UTC")
    
    def start_scheduler(self):
        """Inicia o agendador em background"""
        self.scheduler_active = True
        logger.info("🚀 Sistema de automação iniciado")
        
        while self.scheduler_active:
            schedule.run_pending()
            time.sleep(60)  # Verificar a cada minuto
    
    def stop_scheduler(self):
        """Para o agendador"""
        self.scheduler_active = False
        logger.info("⏹️ Sistema de automação parado")

def main():
    """Função principal do sistema"""
    print("\n" + "="*60)
    print("🌤️ SISTEMA DE PREVISÃO CLIMÁTICA - NOMADS/GFS")
    print("="*60 + "\n")
    
    # Criar gerenciador
    manager = WeatherDataManager()
    
    # Menu interativo
    while True:
        print("\n📋 MENU PRINCIPAL:")
        print("1. 📥 Baixar dados mais recentes")
        print("2. 🔄 Executar atualização completa")
        print("3. 📊 Gerar relatório dos dados")
        print("4. 🤖 Iniciar automação")
        print("5. ⚙️ Configurações")
        print("6. 🔌 Integrar com modelo preditivo")
        print("0. ❌ Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == '1':
            print("\n📥 Baixando dados...")
            result = manager.download_gfs_data()
            if result['success']:
                print(f"✅ {len(result['files'])} arquivos baixados")
            else:
                print("❌ Erro no download")
        
        elif choice == '2':
            print("\n🔄 Executando atualização completa...")
            success = manager.run_automatic_update()
            if success:
                print("✅ Atualização concluída!")
            else:
                print("❌ Erro na atualização")
        
        elif choice == '3':
            print("\n📊 Gerando relatório...")
            # Carregar dados mais recentes
            latest_file = manager.processed_dir / "latest_weather_data.csv"
            if latest_file.exists():
                df = pd.read_csv(latest_file)
                manager.generate_report(df)
            else:
                print("❌ Nenhum dado disponível")
        
        elif choice == '4':
            print("\n🤖 Iniciando sistema de automação...")
            automation = WeatherAutomation(manager)
            automation.schedule_updates()
            print("⏰ Atualizações agendadas. Pressione Ctrl+C para parar.")
            try:
                automation.start_scheduler()
            except KeyboardInterrupt:
                automation.stop_scheduler()
                print("\n⏹️ Automação interrompida")
        
        elif choice == '5':
            print("\n⚙️ Configurações atuais:")
            print(json.dumps(manager.config, indent=2))
            
            if input("\nDeseja editar? (s/n): ").lower() == 's':
                config_file = 'config/weather_config.json'
                print(f"📝 Edite o arquivo: {config_file}")
                if sys.platform == 'win32':
                    os.system(f'notepad {config_file}')
                else:
                    os.system(f'nano {config_file}')
                manager.config = manager.load_config(config_file)
        
        elif choice == '6':
            print("\n🔌 Integrando com modelo preditivo...")
            sales_file = input("Caminho do arquivo de vendas (Enter para padrão): ").strip()
            if not sales_file:
                sales_file = "data/datasets/Loja1_dados_unificados.csv"
            
            if os.path.exists(sales_file):
                latest_weather = manager.processed_dir / "latest_weather_data.csv"
                if latest_weather.exists():
                    weather_df = pd.read_csv(latest_weather)
                    merged = manager.merge_with_sales_data(weather_df, sales_file)
                    if merged is not None:
                        print("✅ Integração concluída!")
                        print(f"📊 Total de registros: {len(merged)}")
                else:
                    print("❌ Nenhum dado meteorológico disponível")
            else:
                print(f"❌ Arquivo não encontrado: {sales_file}")
        
        elif choice == '0':
            print("\n👋 Encerrando sistema...")
            break
        
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        print(f"\n❌ Erro fatal: {e}")
        print("Verifique o arquivo de log para mais detalhes.")