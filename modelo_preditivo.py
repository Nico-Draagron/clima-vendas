# ============================================================================
# 🤖 modelo_preditivo.py - BACKEND DO MODELO PREDITIVO
# ============================================================================
try:
    from modelo_preditivo_bridge import ModeloPreditivoUnificado
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModeloVendasBootstrap:
    """
    Modelo de previsão de vendas usando Random Forest com Bootstrap Resampling
    """
    
    def __init__(self, n_bootstrap_samples=100, test_size=0.2, random_state=42):
        """
        Inicializa o modelo com parâmetros de bootstrap
        
        Args:
            n_bootstrap_samples: Número de amostras bootstrap para gerar
            test_size: Proporção dos dados para teste
            random_state: Seed para reprodutibilidade
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.test_size = test_size
        self.random_state = random_state
        self.models = []
        self.scalers = []
        self.feature_names = None
        self.is_trained = False
        
    def preparar_dados(self, df, target_col, feature_cols=None):
        """
        Prepara os dados para treinamento
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (vendas)
            feature_cols: Lista de colunas de features (se None, usa padrão)
        
        Returns:
            X: Features preparadas
            y: Target
        """
        # Se não especificar features, usar padrão
        if feature_cols is None:
            feature_cols = self._selecionar_features_automaticas(df, target_col)
        
        # Remover linhas com valores faltantes
        df_clean = df[feature_cols + [target_col]].dropna()
        
        # Separar features e target
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Guardar nomes das features
        self.feature_names = feature_cols
        
        return X, y
    
    def _selecionar_features_automaticas(self, df, target_col):
        """
        Seleciona features automaticamente baseado nos dados disponíveis
        """
        features = []
        
        # Features climáticas
        clima_features = [
            'temp_max', 'temp_min', 'temp_media',
            'umid_max', 'umid_min', 'umid_mediana',
            'rad_min', 'rad_max', 'rad_mediana',
            'vento_raj_max', 'vento_vel_media',
            'precipitacao_total'
        ]
        
        for feat in clima_features:
            if feat in df.columns:
                features.append(feat)
        
        # Features temporais (se existirem)
        if 'data' in df.columns:
            df_temp = df.copy()
            df_temp['data'] = pd.to_datetime(df_temp['data'])
            
            # Adicionar features temporais
            df_temp['dia_semana'] = df_temp['data'].dt.dayofweek
            df_temp['dia_mes'] = df_temp['data'].dt.day
            df_temp['mes'] = df_temp['data'].dt.month
            
            if 'dia_semana' not in df.columns:
                df['dia_semana'] = df_temp['dia_semana']
            if 'dia_mes' not in df.columns:
                df['dia_mes'] = df_temp['dia_mes']
            if 'mes' not in df.columns:
                df['mes'] = df_temp['mes']
            
            features.extend(['dia_semana', 'dia_mes', 'mes'])
        
        return features
    
    def treinar(self, X, y, params_rf=None):
        """
        Treina múltiplos modelos usando bootstrap resampling
        
        Args:
            X: Features
            y: Target
            params_rf: Parâmetros para o RandomForest
        
        Returns:
            dict: Relatório de treinamento
        """
        if params_rf is None:
            params_rf = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
        
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Treinar múltiplos modelos com bootstrap
        self.models = []
        self.scalers = []
        train_scores = []
        test_scores = []
        
        for i in range(self.n_bootstrap_samples):
            # Bootstrap sampling
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices]
            
            # Normalizar dados
            scaler = StandardScaler()
            X_bootstrap_scaled = scaler.fit_transform(X_bootstrap)
            X_test_scaled = scaler.transform(X_test)
            
            # Treinar modelo
            model = RandomForestRegressor(**params_rf)
            model.fit(X_bootstrap_scaled, y_bootstrap)
            
            # Guardar modelo e scaler
            self.models.append(model)
            self.scalers.append(scaler)
            
            # Avaliar
            train_score = model.score(X_bootstrap_scaled, y_bootstrap)
            test_score = model.score(X_test_scaled, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        self.is_trained = True
        
        # Calcular métricas agregadas
        y_pred_ensemble = self.prever(X_test, usar_ensemble=True, retornar_intervalo=False)['predicao']
        
        metricas = {
            'RMSE': {
                'valor': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
                'media': np.mean([np.sqrt(mean_squared_error(y_test, 
                         self.models[i].predict(self.scalers[i].transform(X_test))))
                         for i in range(len(self.models))])
            },
            'R²': {
                'valor': r2_score(y_test, y_pred_ensemble),
                'media': np.mean(test_scores)
            },
            'MAE': {
                'valor': mean_absolute_error(y_test, y_pred_ensemble)
            }
        }
        
        # Importância das features
        importancia_features = self._calcular_importancia_features()
        
        return {
            'metricas': metricas,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'n_bootstrap_samples': self.n_bootstrap_samples,
            'importancia_features': importancia_features,
            'melhor_modelo': f"Modelo {np.argmax(test_scores) + 1}"
        }
    
    def prever(self, X, usar_ensemble=True, retornar_intervalo=True, confianca=0.95):
        """
        Faz predições usando os modelos treinados
        
        Args:
            X: Features para predição
            usar_ensemble: Se True, usa média de todos os modelos
            retornar_intervalo: Se True, retorna intervalo de confiança
            confianca: Nível de confiança para o intervalo
        
        Returns:
            dict: Predições e intervalos de confiança
        """
        if not self.is_trained:
            raise ValueError("Modelo não treinado. Execute treinar() primeiro.")
        
        predicoes = []
        
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predicoes.append(pred)
        
        predicoes = np.array(predicoes)
        
        if usar_ensemble:
            predicao_final = np.mean(predicoes, axis=0)
        else:
            # Usar melhor modelo individual
            predicao_final = predicoes[0]
        
        resultado = {'predicao': predicao_final}
        
        if retornar_intervalo:
            # Calcular intervalo de confiança
            std_pred = np.std(predicoes, axis=0)
            z_score = 1.96 if confianca == 0.95 else 2.58  # 95% ou 99%
            
            resultado['intervalo_inferior'] = predicao_final - z_score * std_pred
            resultado['intervalo_superior'] = predicao_final + z_score * std_pred
            resultado['desvio_padrao'] = std_pred
        
        return resultado
    
    def _calcular_importancia_features(self):
        """
        Calcula a importância média das features
        """
        if not self.is_trained:
            return {}
        
        importancias = []
        
        for model in self.models:
            importancias.append(model.feature_importances_)
        
        importancia_media = np.mean(importancias, axis=0)
        importancia_std = np.std(importancias, axis=0)
        
        resultado = {}
        for i, nome in enumerate(self.feature_names):
            resultado[nome] = {
                'importancia': float(importancia_media[i]),
                'desvio': float(importancia_std[i])
            }
        
        # Ordenar por importância
        resultado = dict(sorted(resultado.items(), 
                              key=lambda x: x[1]['importancia'], 
                              reverse=True))
        
        return resultado
    
    def validacao_temporal(self, X, y, n_splits=5):
        """
        Realiza validação temporal (time series split)
        
        Args:
            X: Features
            y: Target
            n_splits: Número de splits temporais
        
        Returns:
            dict: Resultados da validação
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Treinar modelo temporário
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
            model.fit(X_train_scaled, y_train)
            
            score = model.score(X_test_scaled, y_test)
            scores.append(score)
        
        return {
            'scores': scores,
            'media': np.mean(scores),
            'desvio': np.std(scores)
        }
    
    def analise_residuos(self, X, y):
        """
        Analisa os resíduos do modelo
        
        Args:
            X: Features
            y: Target real
        
        Returns:
            dict: Análise dos resíduos
        """
        if not self.is_trained:
            raise ValueError("Modelo não treinado.")
        
        predicoes = self.prever(X, usar_ensemble=True, retornar_intervalo=False)['predicao']
        residuos = y - predicoes
        
        return {
            'residuos': residuos,
            'media': float(np.mean(residuos)),
            'desvio': float(np.std(residuos)),
            'min': float(np.min(residuos)),
            'max': float(np.max(residuos)),
            'mape': float(np.mean(np.abs(residuos / y)) * 100)
        }

# ============================================================================
# 🔧 FUNÇÕES AUXILIARES
# ============================================================================

def criar_modelo_simples():
    """
    Cria uma instância simples do modelo para testes
    """
    return ModeloVendasBootstrap(
        n_bootstrap_samples=50,
        test_size=0.2,
        random_state=42
    )

def testar_modelo_com_dados_exemplo():
    """
    Testa o modelo com dados sintéticos
    """
    # Criar dados sintéticos
    np.random.seed(42)
    n_samples = 365
    
    df = pd.DataFrame({
        'data': pd.date_range('2023-01-01', periods=n_samples),
        'temp_media': np.random.normal(25, 5, n_samples),
        'precipitacao_total': np.random.exponential(5, n_samples),
        'umidade_media': np.random.normal(70, 10, n_samples),
        'valor_vendas': np.random.normal(5000, 1000, n_samples)
    })
    
    # Adicionar correlação com temperatura
    df['valor_vendas'] += df['temp_media'] * 50
    
    # Criar e treinar modelo
    modelo = criar_modelo_simples()
    
    # Preparar dados
    X, y = modelo.preparar_dados(
        df, 
        target_col='valor_vendas',
        feature_cols=['temp_media', 'precipitacao_total', 'umidade_media']
    )
    
    # Treinar
    relatorio = modelo.treinar(X, y)
    
    return modelo, relatorio

# ============================================================================
# 🚀 TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    print("🤖 Testando Modelo Preditivo...")
    
    try:
        modelo, relatorio = testar_modelo_com_dados_exemplo()
        
        print("\n✅ Modelo treinado com sucesso!")
        print(f"📊 R² médio: {relatorio['metricas']['R²']['media']:.3f}")
        print(f"📈 RMSE: {relatorio['metricas']['RMSE']['valor']:.2f}")
        
        print("\n🎯 Importância das Features:")
        for feat, imp in relatorio['importancia_features'].items():
            print(f"  - {feat}: {imp['importancia']:.3f}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")