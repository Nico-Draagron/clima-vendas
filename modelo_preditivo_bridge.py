# ============================================================================
# 🌉 modelo_preditivo_bridge.py - PONTE ENTRE SISTEMA ANTIGO E NOVO
# ============================================================================
# Este arquivo mantém compatibilidade com o código existente enquanto
# adiciona as novas funcionalidades do sistema NOMADS

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Tentar importar o modelo avançado primeiro
try:
    from modelo_preditivo_integrado import ModeloVendasClimaticoAvancado
    ADVANCED_MODEL_AVAILABLE = True
except ImportError:
    ADVANCED_MODEL_AVAILABLE = False
    print("⚠️ Modelo avançado não disponível. Usando modelo básico.")

# Importar modelo básico como fallback
try:
    from modelo_preditivo import ModeloVendasBootstrap
    BASIC_MODEL_AVAILABLE = True
except ImportError:
    BASIC_MODEL_AVAILABLE = False

class ModeloPreditivoUnificado:
    """
    Classe unificada que escolhe automaticamente entre modelo básico e avançado
    """
    
    def __init__(self, use_advanced=None):
        """
        Inicializa o modelo apropriado
        
        Args:
            use_advanced: None (auto), True (forçar avançado), False (forçar básico)
        """
        self.mode = 'none'
        self.model = None
        
        # Determinar qual modelo usar
        if use_advanced is None:
            # Auto-detectar
            if ADVANCED_MODEL_AVAILABLE:
                self._init_advanced_model()
            elif BASIC_MODEL_AVAILABLE:
                self._init_basic_model()
            else:
                raise ImportError("Nenhum modelo disponível!")
        elif use_advanced:
            if ADVANCED_MODEL_AVAILABLE:
                self._init_advanced_model()
            else:
                raise ImportError("Modelo avançado não disponível!")
        else:
            if BASIC_MODEL_AVAILABLE:
                self._init_basic_model()
            else:
                raise ImportError("Modelo básico não disponível!")
    
    def _init_advanced_model(self):
        """Inicializa modelo avançado com NOMADS"""
        print("🚀 Usando Modelo Avançado com integração NOMADS")
        self.model = ModeloVendasClimaticoAvancado()
        self.mode = 'advanced'
    
    def _init_basic_model(self):
        """Inicializa modelo básico"""
        print("📊 Usando Modelo Básico")
        self.model = ModeloVendasBootstrap()
        self.mode = 'basic'
    
    def check_nomads_data(self):
        """Verifica se há dados NOMADS disponíveis"""
        nomads_dir = "processed_data"
        latest_file = os.path.join(nomads_dir, "latest_weather_data.csv")
        
        if os.path.exists(latest_file):
            # Verificar idade do arquivo
            import time
            file_age = time.time() - os.path.getmtime(latest_file)
            hours_old = file_age / 3600
            
            if hours_old < 24:
                return {
                    'available': True,
                    'path': latest_file,
                    'age_hours': hours_old,
                    'status': 'current'
                }
            else:
                return {
                    'available': True,
                    'path': latest_file,
                    'age_hours': hours_old,
                    'status': 'outdated'
                }
        return {
            'available': False,
            'path': None,
            'age_hours': None,
            'status': 'missing'
        }
    
    def update_weather_data(self):
        """Atualiza dados meteorológicos do NOMADS"""
        if self.mode == 'advanced':
            try:
                from sistema_previsao_climatica import WeatherDataManager
                
                print("📥 Atualizando dados meteorológicos...")
                manager = WeatherDataManager()
                result = manager.run_automatic_update()
                
                if result:
                    print("✅ Dados atualizados com sucesso!")
                    return True
                else:
                    print("❌ Falha na atualização")
                    return False
            except Exception as e:
                print(f"❌ Erro ao atualizar: {e}")
                return False
        else:
            print("⚠️ Atualização NOMADS disponível apenas no modo avançado")
            return False
    
    def prepare_data(self, df=None, source='auto'):
        """
        Prepara dados para treinamento
        
        Args:
            df: DataFrame com dados (opcional)
            source: 'auto', 'csv', 'nomads'
        """
        if source == 'auto':
            # Tentar NOMADS primeiro
            nomads_status = self.check_nomads_data()
            if nomads_status['available']:
                if nomads_status['status'] == 'outdated':
                    print(f"⚠️ Dados NOMADS com {nomads_status['age_hours']:.1f} horas")
                    if input("Atualizar? (s/n): ").lower() == 's':
                        self.update_weather_data()
                
                if self.mode == 'advanced':
                    return self.model.load_and_prepare_data(nomads_status['path'])
            
            # Fallback para CSV
            if df is None:
                csv_path = "data/datasets/Loja1_dados_unificados.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['data'] = pd.to_datetime(df['data'])
                else:
                    raise FileNotFoundError("Nenhum dado disponível")
        
        elif source == 'nomads':
            if self.mode == 'advanced':
                nomads_status = self.check_nomads_data()
                if nomads_status['available']:
                    return self.model.load_and_prepare_data(nomads_status['path'])
                else:
                    raise FileNotFoundError("Dados NOMADS não disponíveis")
            else:
                raise ValueError("NOMADS requer modo avançado")
        
        elif source == 'csv':
            if df is None:
                csv_path = "data/datasets/Loja1_dados_unificados.csv"
                df = pd.read_csv(csv_path)
                df['data'] = pd.to_datetime(df['data'])
        
        # Processar dados conforme o modo
        if self.mode == 'advanced':
            return self.model.engineer_features(df)
        else:
            # Preparar para modelo básico
            target_col = 'valor_loja_01'
            feature_cols = [col for col in df.columns 
                          if col not in ['data', target_col] and 
                          not col.endswith('_pct')]
            
            X = df[feature_cols].fillna(0)
            y = df[target_col] if target_col in df else None
            
            return X, y
    
    def train(self, df=None, target_col='valor_loja_01', **kwargs):
        """
        Treina o modelo
        
        Args:
            df: DataFrame com dados
            target_col: Coluna alvo
            **kwargs: Parâmetros adicionais
        """
        if self.mode == 'advanced':
            # Modelo avançado
            if df is None:
                df = self.prepare_data(source='auto')
            
            return self.model.train(df, target_col)
        
        elif self.mode == 'basic':
            # Modelo básico
            if df is None:
                X, y = self.prepare_data(source='auto')
            else:
                X, y = self.prepare_data(df, source='csv')
            
            if y is None:
                raise ValueError(f"Coluna {target_col} não encontrada")
            
            return self.model.treinar(X, y)
        
        else:
            raise RuntimeError("Nenhum modelo inicializado")
    
    def predict(self, X, **kwargs):
        """
        Faz previsões
        
        Args:
            X: Features para previsão
            **kwargs: Parâmetros adicionais
        """
        if self.mode == 'advanced':
            # Verificar se quer intervalo de confiança
            if kwargs.get('with_confidence', False):
                return self.model.predict_with_confidence(X, kwargs.get('confidence', 0.95))
            else:
                return self.model.predict(X)
        
        elif self.mode == 'basic':
            # Modelo básico
            if kwargs.get('with_confidence', False):
                return self.model.prever(X, usar_ensemble=True, retornar_intervalo=True)
            else:
                return self.model.prever(X, usar_ensemble=True, retornar_intervalo=False)
        
        else:
            raise RuntimeError("Modelo não treinado")
    
    def save(self, path=None):
        """Salva o modelo"""
        if path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            path = f"models/modelo_{self.mode}_{timestamp}.pkl"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.mode == 'advanced':
            self.model.save_model(path)
        elif self.mode == 'basic':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        
        print(f"✅ Modelo salvo em: {path}")
        return path
    
    def load(self, path):
        """Carrega modelo salvo"""
        if self.mode == 'advanced':
            self.model.load_model(path)
        elif self.mode == 'basic':
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        
        print(f"✅ Modelo carregado de: {path}")
    
    def get_feature_importance(self):
        """Retorna importância das features"""
        if self.mode == 'advanced':
            if hasattr(self.model, 'feature_importance'):
                return self.model.feature_importance
        elif self.mode == 'basic':
            if hasattr(self.model, 'modelos') and self.model.modelos:
                # Pegar do Random Forest se disponível
                for modelo in self.model.modelos:
                    if hasattr(modelo, 'feature_importances_'):
                        return pd.DataFrame({
                            'feature': self.model.feature_names,
                            'importance': modelo.feature_importances_
                        }).sort_values('importance', ascending=False)
        return None
    
    def generate_report(self):
        """Gera relatório do modelo"""
        report = []
        report.append("="*60)
        report.append(f"📊 RELATÓRIO DO MODELO - Modo: {self.mode.upper()}")
        report.append("="*60)
        
        # Status dos dados
        nomads_status = self.check_nomads_data()
        report.append("\n📡 Status dos Dados NOMADS:")
        if nomads_status['available']:
            report.append(f"  ✅ Disponível")
            report.append(f"  📅 Idade: {nomads_status['age_hours']:.1f} horas")
            report.append(f"  📊 Status: {nomads_status['status']}")
        else:
            report.append("  ❌ Não disponível")
        
        # Informações do modelo
        if self.mode == 'advanced' and hasattr(self.model, 'metrics_history'):
            if self.model.metrics_history:
                latest = self.model.metrics_history[-1]
                report.append(f"\n🏆 Melhor Modelo: {latest.get('best_model', 'N/A')}")
                
                if 'results' in latest:
                    report.append("\n📈 Métricas:")
                    for model_name, metrics in latest['results'].items():
                        report.append(f"  {model_name}:")
                        report.append(f"    R²: {metrics.get('test_r2', 0):.3f}")
                        report.append(f"    RMSE: {metrics.get('test_rmse', 0):.2f}")
        
        elif self.mode == 'basic':
            report.append("\n📊 Modelo Básico Ativo")
            report.append("  Algoritmo: Random Forest com Bootstrap")
        
        # Features importantes
        importance = self.get_feature_importance()
        if importance is not None:
            report.append("\n🎯 Top 5 Features:")
            for idx, row in importance.head(5).iterrows():
                report.append(f"  {idx+1}. {row['feature']}: {row['importance']:.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)

# Função de compatibilidade para código legado
def get_modelo():
    """Retorna instância do modelo unificado (compatibilidade)"""
    return ModeloPreditivoUnificado()

# CLI para teste rápido
if __name__ == "__main__":
    print("\n🌉 TESTE DO MODELO UNIFICADO")
    print("="*50)
    
    # Criar modelo
    modelo = ModeloPreditivoUnificado()
    
    # Verificar status
    print("\n📊 Status do Sistema:")
    print(f"  Modo: {modelo.mode}")
    print(f"  Modelo Avançado: {'✅' if ADVANCED_MODEL_AVAILABLE else '❌'}")
    print(f"  Modelo Básico: {'✅' if BASIC_MODEL_AVAILABLE else '❌'}")
    
    # Verificar NOMADS
    nomads = modelo.check_nomads_data()
    print(f"\n📡 NOMADS:")
    print(f"  Disponível: {'✅' if nomads['available'] else '❌'}")
    if nomads['available']:
        print(f"  Idade: {nomads['age_hours']:.1f} horas")
        print(f"  Status: {nomads['status']}")
    
    # Menu
    while True:
        print("\n📋 Opções:")
        print("1. Atualizar dados NOMADS")
        print("2. Treinar modelo")
        print("3. Gerar relatório")
        print("0. Sair")
        
        choice = input("\nEscolha: ").strip()
        
        if choice == '1':
            modelo.update_weather_data()
        elif choice == '2':
            print("Preparando dados...")
            try:
                if modelo.mode == 'advanced':
                    df = modelo.prepare_data(source='auto')
                    print("Treinando modelo avançado...")
                    modelo.train(df)
                else:
                    X, y = modelo.prepare_data(source='auto')
                    print("Treinando modelo básico...")
                    modelo.train()
                print("✅ Treinamento concluído!")
            except Exception as e:
                print(f"❌ Erro: {e}")
        elif choice == '3':
            print(modelo.generate_report())
        elif choice == '0':
            break