import pandas as pd
import streamlit as st
import os
from typing import Optional, Dict, List

class DatasetManager:
    """Gerenciador para carregamento e validação de datasets"""
    
    def __init__(self):
        self.datasets_path = 'data/datasets'
        self.ensure_datasets_folder()
    
    def ensure_datasets_folder(self):
        """Garante que a pasta de datasets existe"""
        if not os.path.exists(self.datasets_path):
            os.makedirs(self.datasets_path, exist_ok=True)
    
    @st.cache_data
    def load_dataset(_self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Carrega dataset com cache para performance"""
        try:
            file_path = os.path.join(_self.datasets_path, dataset_name)
            
            if not os.path.exists(file_path):
                st.error(f"Dataset '{dataset_name}' não encontrado")
                return None
            
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    
                    # Validações básicas
                    if df.empty:
                        st.warning(f"Dataset '{dataset_name}' está vazio")
                        return None
                    
                    # Processamento padrão de datas se existir coluna 'data'
                    if 'data' in df.columns:
                        df['data'] = pd.to_datetime(df['data'], errors='coerce')
                    
                    return df
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Erro ao processar dataset: {e}")
                    return None
            
            st.error(f"Não foi possível decodificar o arquivo '{dataset_name}'")
            return None
            
        except Exception as e:
            st.error(f"Erro ao carregar dataset '{dataset_name}': {e}")
            return None
    
    def list_available_datasets(self) -> List[str]:
        """Lista todos os datasets disponíveis"""
        if not os.path.exists(self.datasets_path):
            return []
        
        datasets = []
        for file in os.listdir(self.datasets_path):
            if file.endswith('.csv'):
                datasets.append(file)
        
        return sorted(datasets)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Retorna informações sobre um dataset"""
        try:
            file_path = os.path.join(self.datasets_path, dataset_name)
            
            if not os.path.exists(file_path):
                return {"error": "Arquivo não encontrado"}
            
            # Informações do arquivo
            file_size = os.path.getsize(file_path)
            
            # Carregar apenas primeiras linhas para análise
            df_sample = pd.read_csv(file_path, nrows=100)
            
            info = {
                "nome": dataset_name,
                "tamanho_arquivo": f"{file_size / 1024:.1f} KB",
                "colunas": len(df_sample.columns),
                "tipos_dados": df_sample.dtypes.to_dict(),
                "colunas_lista": list(df_sample.columns),
                "primeiras_linhas": df_sample.head(3).to_dict('records')
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Erro ao analisar dataset: {e}"}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str = 'clima_vendas') -> Dict:
        """Valida se dataset tem estrutura esperada"""
        validation = {"valid": True, "warnings": [], "errors": []}
        
        if dataset_type == 'clima_vendas':
            # Colunas esperadas para análise clima x vendas
            required_columns = ['data']
            recommended_columns = [
                'valor_total', 'temp_max', 'temp_min', 'temp_media',
                'precipitacao_total', 'umid_mediana'
            ]
            
            # Verificar colunas obrigatórias
            for col in required_columns:
                if col not in df.columns:
                    validation['errors'].append(f"Coluna obrigatória '{col}' não encontrada")
                    validation['valid'] = False
            
            # Verificar colunas recomendadas
            for col in recommended_columns:
                if col not in df.columns:
                    validation['warnings'].append(f"Coluna recomendada '{col}' não encontrada")
        
        # Verificações gerais
        if len(df) == 0:
            validation['errors'].append("Dataset vazio")
            validation['valid'] = False
        
        if len(df.columns) == 0:
            validation['errors'].append("Nenhuma coluna encontrada")
            validation['valid'] = False
        
        return validation