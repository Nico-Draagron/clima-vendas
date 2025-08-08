# ============================================================================
# 🤖 modelo_preditivo_integrado.py - MODELO PREDITIVO COM DADOS REAIS NOMADS
# ============================================================================
# Versão melhorada do modelo preditivo com integração completa ao NOMADS/GFS

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
import joblib
import pickle
from datetime import datetime, timedelta
import os
import json
import logging

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModeloVendasClimaticoAvancado:
    """
    Modelo avançado de previsão de vendas com dados climáticos reais NOMADS
    """
    
    def __init__(self, config_path='config/model_config.json'):
        """Inicializa o modelo com configurações avançadas"""
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.best_model = None
        self.feature_importance = None
        self.metrics_history = []
        self.is_trained = False
        
        # Configurar modelos disponíveis
        self.initialize_models()
    
    def load_config(self, config_path):
        """Carrega configurações do modelo"""
        default_config = {
            'models': {
                'random_forest': {
                    'enabled': True,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 15,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'random_state': 42
                    }
                },
                'gradient_boosting': {
                    'enabled': True,
                    'params': {
                        'n_estimators': 150,
                        'max_depth': 7,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                },
                'extra_trees': {
                    'enabled': True,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 20,
                        'random_state': 42
                    }
                },
                'svr': {
                    'enabled': True,
                    'params': {
                        'kernel': 'rbf',
                        'C': 100,
                        'gamma': 'scale'
                    }
                },
                'linear': {
                    'enabled': True,
                    'params': {}
                }
            },
            'features': {
                'weather_features': [
                    'temp_media', 'temp_max', 'temp_min',
                    'precipitacao_total', 'umid_mediana',
                    'pressao_media', 'vento_vel_media',
                    'rad_mediana', 'nuvens_media'
                ],
                'engineered_features': [
                    'temp_range', 'temp_squared', 'feels_like',
                    'humidity_temp_interaction', 'rain_category',
                    'wind_chill', 'heat_index', 'weather_score'
                ],
                'lag_features': {
                    'enabled': True,
                    'lags': [1, 7, 14, 30]
                },
                'rolling_features': {
                    'enabled': True,
                    'windows': [3, 7, 14, 30]
                }
            },
            'preprocessing': {
                'scaler': 'robust',  # 'standard', 'robust', 'minmax'
                'handle_outliers': True,
                'outlier_threshold': 3,
                'fill_missing': 'interpolate'  # 'mean', 'median', 'forward', 'interpolate'
            },
            'validation': {
                'test_size': 0.2,
                'cv_folds': 5,
                'time_series_split': True,
                'bootstrap_samples': 100
            },
            'feature_selection': {
                'enabled': True,
                'method': 'mutual_info',  # 'f_score', 'mutual_info', 'rfe'
                'top_k': 20
            }
        }
        
        # Tentar carregar configuração personalizada
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                # Merge recursivo das configurações
                self._merge_configs(default_config, custom_config)
                logger.info(f"✅ Configuração personalizada carregada de {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar config: {e}. Usando padrão.")
        else:
            # Criar arquivo de configuração
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"📝 Arquivo de configuração criado: {config_path}")
        
        return default_config
    
    def _merge_configs(self, default, custom):
        """Merge recursivo de configurações"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def initialize_models(self):
        """Inicializa os modelos configurados"""
        model_configs = self.config['models']
        
        if model_configs['random_forest']['enabled']:
            self.models['RandomForest'] = RandomForestRegressor(
                **model_configs['random_forest']['params']
            )
        
        if model_configs['gradient_boosting']['enabled']:
            self.models['GradientBoosting'] = GradientBoostingRegressor(
                **model_configs['gradient_boosting']['params']
            )
        
        if model_configs['extra_trees']['enabled']:
            self.models['ExtraTrees'] = ExtraTreesRegressor(
                **model_configs['extra_trees']['params']
            )
        
        if model_configs['svr']['enabled']:
            self.models['SVR'] = SVR(
                **model_configs['svr']['params']
            )
        
        if model_configs['linear']['enabled']:
            self.models['Linear'] = LinearRegression(
                **model_configs['linear']['params']
            )
            self.models['Ridge'] = Ridge(alpha=1.0, random_state=42)
            self.models['Lasso'] = Lasso(alpha=0.1, random_state=42)
        
        logger.info(f"✅ {len(self.models)} modelos inicializados")
    
    def load_and_prepare_data(self, data_path=None):
        """Carrega e prepara dados com features avançadas"""
        
        # Usar caminho padrão se não especificado
        if data_path is None:
            data_path = "processed_data/latest_weather_data.csv"
        
        logger.info(f"📂 Carregando dados de {data_path}")
        
        try:
            # Carregar dados
            df = pd.read_csv(data_path)
            df['data'] = pd.to_datetime(df['data'])
            
            # Ordenar por data
            df = df.sort_values('data').reset_index(drop=True)
            
            logger.info(f"✅ {len(df)} registros carregados")
            
            # Aplicar engenharia de features
            df = self.engineer_features(df)
            
            # Tratar valores faltantes
            df = self.handle_missing_values(df)
            
            # Remover outliers se configurado
            if self.config['preprocessing']['handle_outliers']:
                df = self.remove_outliers(df)
            
            logger.info(f"✅ Dados preparados: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def engineer_features(self, df):
        """Cria features avançadas de engenharia"""
        logger.info("🔧 Criando features de engenharia...")
        
        # Features básicas de temperatura
        if 'temp_max' in df and 'temp_min' in df:
            df['temp_range'] = df['temp_max'] - df['temp_min']
        
        if 'temp_media' in df:
            df['temp_squared'] = df['temp_media'] ** 2
            df['temp_cubed'] = df['temp_media'] ** 3
        
        # Sensação térmica
        if 'temp_media' in df and 'umid_mediana' in df:
            df['feels_like'] = self.calculate_feels_like(
                df['temp_media'], 
                df['umid_mediana'],
                df.get('vento_vel_media', 0)
            )
        
        # Interações entre variáveis
        if 'umid_mediana' in df and 'temp_media' in df:
            df['humidity_temp_interaction'] = df['umid_mediana'] * df['temp_media']
        
        # Categorias de chuva
        if 'precipitacao_total' in df:
            df['rain_category'] = pd.cut(
                df['precipitacao_total'],
                bins=[-0.1, 0.1, 5, 20, 100],
                labels=[0, 1, 2, 3]  # sem chuva, leve, moderada, forte
            ).astype(float)
        
        # Wind chill (sensação de frio com vento)
        if 'temp_media' in df and 'vento_vel_media' in df:
            df['wind_chill'] = self.calculate_wind_chill(
                df['temp_media'],
                df['vento_vel_media']
            )
        
        # Heat index (índice de calor)
        if 'temp_media' in df and 'umid_mediana' in df:
            df['heat_index'] = self.calculate_heat_index(
                df['temp_media'],
                df['umid_mediana']
            )
        
        # Score composto de condições climáticas
        df['weather_score'] = self.calculate_weather_score(df)
        
        # Features temporais
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_mes'] = df['data'].dt.day
        df['mes'] = df['data'].dt.month
        df['trimestre'] = df['data'].dt.quarter
        df['dia_ano'] = df['data'].dt.dayofyear
        df['semana_ano'] = df['data'].dt.isocalendar().week
        
        # Features cíclicas (seno e cosseno para capturar periodicidade)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        # Criar lag features se configurado
        if self.config['features']['lag_features']['enabled']:
            df = self.create_lag_features(df)
        
        # Criar rolling features se configurado
        if self.config['features']['rolling_features']['enabled']:
            df = self.create_rolling_features(df)
        
        logger.info(f"✅ Features criadas: {df.shape[1]} colunas")
        
        return df
    
    def calculate_feels_like(self, temp, humidity, wind_speed):
        """Calcula sensação térmica"""
        # Fórmula simplificada de sensação térmica
        feels_like = temp
        
        # Ajuste para umidade alta
        if isinstance(temp, pd.Series):
            mask = (temp > 27) & (humidity > 40)
            feels_like[mask] = temp[mask] + (humidity[mask] - 40) * 0.1
        else:
            if temp > 27 and humidity > 40:
                feels_like = temp + (humidity - 40) * 0.1
        
        # Ajuste para vento
        if isinstance(wind_speed, pd.Series):
            feels_like = feels_like - wind_speed * 0.5
        else:
            feels_like = feels_like - wind_speed * 0.5
        
        return feels_like
    
    def calculate_wind_chill(self, temp, wind_speed):
        """Calcula wind chill (sensação de frio com vento)"""
        # Fórmula do wind chill (simplificada)
        if isinstance(temp, pd.Series):
            wind_chill = temp.copy()
            mask = (temp < 10) & (wind_speed > 5)
            wind_chill[mask] = 13.12 + 0.6215 * temp[mask] - 11.37 * (wind_speed[mask] ** 0.16) + 0.3965 * temp[mask] * (wind_speed[mask] ** 0.16)
        else:
            if temp < 10 and wind_speed > 5:
                wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
            else:
                wind_chill = temp
        
        return wind_chill
    
    def calculate_heat_index(self, temp, humidity):
        """Calcula índice de calor"""
        # Fórmula do heat index (simplificada)
        if isinstance(temp, pd.Series):
            heat_index = temp.copy()
            mask = (temp > 25) & (humidity > 40)
            c1, c2, c3, c4, c5 = -8.78469475556, 1.61139411, 2.33854883889, -0.14611605, -0.012308094
            heat_index[mask] = c1 + c2*temp[mask] + c3*humidity[mask] + c4*temp[mask]*humidity[mask] + c5*temp[mask]**2
        else:
            if temp > 25 and humidity > 40:
                c1, c2, c3, c4, c5 = -8.78469475556, 1.61139411, 2.33854883889, -0.14611605, -0.012308094
                heat_index = c1 + c2*temp + c3*humidity + c4*temp*humidity + c5*temp**2
            else:
                heat_index = temp
        
        return heat_index
    
    def calculate_weather_score(self, df):
        """Calcula score composto de condições climáticas para vendas"""
        score = pd.Series(100, index=df.index, dtype=float)
        
        # Penalizar temperaturas extremas
        if 'temp_media' in df:
            temp_penalty = np.abs(df['temp_media'] - 22) * 2  # 22°C como ideal
            score -= temp_penalty
        
        # Penalizar chuva
        if 'precipitacao_total' in df:
            rain_penalty = np.minimum(df['precipitacao_total'] * 3, 30)
            score -= rain_penalty
        
        # Penalizar umidade extrema
        if 'umid_mediana' in df:
            humidity_penalty = np.abs(df['umid_mediana'] - 60) * 0.5  # 60% como ideal
            score -= humidity_penalty
        
        # Penalizar vento forte
        if 'vento_vel_media' in df:
            wind_penalty = np.minimum(df['vento_vel_media'] * 2, 20)
            score -= wind_penalty
        
        # Normalizar entre 0 e 100
        score = np.clip(score, 0, 100)
        
        return score
    
    def create_lag_features(self, df):
        """Cria features com lag (valores passados)"""
        lags = self.config['features']['lag_features']['lags']
        
        for col in ['temp_media', 'precipitacao_total', 'umid_mediana', 'weather_score']:
            if col in df:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Features de diferença (mudança em relação ao período anterior)
        for col in ['temp_media', 'precipitacao_total']:
            if col in df:
                df[f'{col}_diff_1'] = df[col].diff(1)
                df[f'{col}_diff_7'] = df[col].diff(7)
        
        return df
    
    def create_rolling_features(self, df):
        """Cria features com janelas móveis"""
        windows = self.config['features']['rolling_features']['windows']
        
        for col in ['temp_media', 'precipitacao_total', 'umid_mediana', 'weather_score']:
            if col in df:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window, min_periods=1).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window, min_periods=1).min()
        
        return df
    
    def handle_missing_values(self, df):
        """Trata valores faltantes de forma inteligente"""
        method = self.config['preprocessing']['fill_missing']
        
        if method == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif method == 'forward':
            df = df.fillna(method='ffill')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        
        # Preencher valores restantes com 0
        df = df.fillna(0)
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers usando z-score"""
        threshold = self.config['preprocessing']['outlier_threshold']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['dia_semana', 'mes', 'dia_mes', 'trimestre']]
        
        for col in numeric_cols:
            if col in df:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        logger.info(f"✅ Outliers removidos. Registros restantes: {len(df)}")
        
        return df
    
    def select_features(self, X, y):
        """Seleciona as melhores features"""
        if not self.config['feature_selection']['enabled']:
            return X
        
        method = self.config['feature_selection']['method']
        k = min(self.config['feature_selection']['top_k'], X.shape[1])
        
        logger.info(f"🎯 Selecionando top {k} features usando {method}")
        
        if method == 'f_score':
            selector = SelectKBest(f_regression, k=k)
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            selector = SelectKBest(mutual_info_regression, k=k)
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
        else:
            return X
        
        X_selected = selector.fit_transform(X, y)
        
        # Salvar nomes das features selecionadas
        selected_indices = selector.get_support(indices=True)
        self.feature_names = X.columns[selected_indices].tolist()
        
        logger.info(f"✅ Features selecionadas: {self.feature_names}")
        
        return pd.DataFrame(X_selected, columns=self.feature_names, index=X.index)
    
    def train(self, df, target_col='valor_loja_01'):
        """Treina todos os modelos e seleciona o melhor"""
        logger.info("🚀 Iniciando treinamento dos modelos...")
        
        # Preparar dados
        feature_cols = [col for col in df.columns if col not in [target_col, 'data']]
        X = df[feature_cols]
        y = df[target_col]
        
        # Remover features com muitos NaN
        X = X.dropna(axis=1, thresh=len(X)*0.5)
        
        # Selecionar features
        X = self.select_features(X, y)
        
        # Dividir dados
        if self.config['validation']['time_series_split']:
            # Para séries temporais, usar split temporal
            split_index = int(len(X) * (1 - self.config['validation']['test_size']))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['validation']['test_size'],
                random_state=42
            )
        
        # Escalar dados
        scaler_type = self.config['preprocessing']['scaler']
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar e avaliar cada modelo
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"📊 Treinando {model_name}...")
            
            try:
                # Treinar modelo
                model.fit(X_train_scaled, y_train)
                
                # Fazer previsões
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calcular métricas
                metrics = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'train_mape': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
                    'test_mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100
                }
                
                # Validação cruzada
                if self.config['validation']['time_series_split']:
                    tscv = TimeSeriesSplit(n_splits=self.config['validation']['cv_folds'])
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, 
                                                scoring='neg_mean_squared_error')
                    metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
                    metrics['cv_rmse_std'] = np.sqrt(-cv_scores).std()
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'scaler': scaler
                }
                
                logger.info(f"  R² Test: {metrics['test_r2']:.3f} | RMSE Test: {metrics['test_rmse']:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao treinar {model_name}: {e}")
                continue
        
        # Selecionar melhor modelo
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['test_r2'])
            self.best_model = results[best_model_name]['model']
            self.scalers['main'] = results[best_model_name]['scaler']
            self.is_trained = True
            
            # Salvar importância das features se disponível
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Salvar histórico de métricas
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_name,
                'results': {k: v['metrics'] for k, v in results.items()}
            })
            
            logger.info(f"🏆 Melhor modelo: {best_model_name}")
            logger.info(f"   R² Test: {results[best_model_name]['metrics']['test_r2']:.3f}")
            logger.info(f"   RMSE Test: {results[best_model_name]['metrics']['test_rmse']:.2f}")
            
            return results
        else:
            logger.error("❌ Nenhum modelo foi treinado com sucesso")
            return None
    
    def predict(self, X):
        """Faz previsões com o melhor modelo"""
        if not self.is_trained:
            raise ValueError("Modelo não está treinado. Execute train() primeiro.")
        
        # Garantir que X tem as mesmas features
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        # Escalar dados
        X_scaled = self.scalers['main'].transform(X)
        
        # Fazer previsão
        predictions = self.best_model.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(self, X, confidence=0.95):
        """Faz previsões com intervalo de confiança usando bootstrap"""
        if not self.is_trained:
            raise ValueError("Modelo não está treinado. Execute train() primeiro.")
        
        n_bootstrap = self.config['validation']['bootstrap_samples']
        predictions = []
        
        # Garantir que X tem as mesmas features
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        # Escalar dados
        X_scaled = self.scalers['main'].transform(X)
        
        # Bootstrap predictions
        for _ in range(n_bootstrap):
            if hasattr(self.best_model, 'estimators_'):
                # Para ensemble models
                estimator = np.random.choice(self.best_model.estimators_)
                pred = estimator.predict(X_scaled)
            else:
                # Para outros modelos, adicionar ruído
                pred = self.best_model.predict(X_scaled)
                noise = np.random.normal(0, pred.std() * 0.1, size=pred.shape)
                pred = pred + noise
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calcular intervalos
        lower = np.percentile(predictions, (1 - confidence) * 100 / 2, axis=0)
        upper = np.percentile(predictions, (1 + confidence) * 100 / 2, axis=0)
        mean_pred = predictions.mean(axis=0)
        
        return {
            'prediction': mean_pred,
            'lower': lower,
            'upper': upper,
            'confidence': confidence
        }
    
    def save_model(self, path='models/modelo_climatico.pkl'):
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo não está treinado")
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        model_data = {
            'best_model': self.best_model,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Modelo salvo em {path}")
    
    def load_model(self, path='models/modelo_climatico.pkl'):
        """Carrega modelo salvo"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        self.metrics_history = model_data['metrics_history']
        self.is_trained = True
        
        logger.info(f"✅ Modelo carregado de {path}")
    
    def generate_report(self):
        """Gera relatório completo do modelo"""
        if not self.is_trained:
            return "Modelo não está treinado"
        
        report = []
        report.append("="*60)
        report.append("📊 RELATÓRIO DO MODELO PREDITIVO CLIMÁTICO")
        report.append("="*60)
        report.append(f"\n📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Últimas métricas
        if self.metrics_history:
            latest = self.metrics_history[-1]
            report.append(f"\n🏆 Melhor Modelo: {latest['best_model']}")
            
            report.append("\n📈 Métricas de Performance:")
            for model_name, metrics in latest['results'].items():
                report.append(f"\n  {model_name}:")
                report.append(f"    R² Test: {metrics['test_r2']:.3f}")
                report.append(f"    RMSE Test: {metrics['test_rmse']:.2f}")
                report.append(f"    MAPE Test: {metrics['test_mape']:.2f}%")
        
        # Top features
        if self.feature_importance is not None:
            report.append("\n🎯 Top 10 Features Mais Importantes:")
            for idx, row in self.feature_importance.head(10).iterrows():
                report.append(f"  {idx+1}. {row['feature']}: {row['importance']:.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)

def main():
    """Função principal para teste e demonstração"""
    print("\n" + "="*60)
    print("🤖 MODELO PREDITIVO CLIMÁTICO AVANÇADO")
    print("="*60 + "\n")
    
    # Criar modelo
    modelo = ModeloVendasClimaticoAvancado()
    
    # Menu interativo
    while True:
        print("\n📋 MENU:")
        print("1. 📂 Carregar e preparar dados")
        print("2. 🚀 Treinar modelo")
        print("3. 🔮 Fazer previsões")
        print("4. 💾 Salvar modelo")
        print("5. 📥 Carregar modelo salvo")
        print("6. 📊 Gerar relatório")
        print("0. ❌ Sair")
        
        choice = input("\nEscolha: ").strip()
        
        if choice == '1':
            data_path = input("Caminho dos dados (Enter para padrão): ").strip()
            df = modelo.load_and_prepare_data(data_path if data_path else None)
            if df is not None:
                print(f"✅ Dados carregados: {df.shape}")
                print(f"📊 Colunas: {list(df.columns)[:10]}...")
        
        elif choice == '2':
            if 'df' not in locals():
                print("❌ Carregue os dados primeiro!")
            else:
                target = input("Coluna alvo (Enter para 'valor_loja_01'): ").strip()
                results = modelo.train(df, target if target else 'valor_loja_01')
                if results:
                    print("✅ Treinamento concluído!")
        
        elif choice == '3':
            if not modelo.is_trained:
                print("❌ Treine o modelo primeiro!")
            else:
                # Exemplo de previsão
                print("🔮 Fazendo previsão para última entrada...")
                last_row = df.iloc[-1:][modelo.feature_names]
                pred = modelo.predict_with_confidence(last_row)
                print(f"📈 Previsão: {pred['prediction'][0]:.2f}")
                print(f"   Intervalo: [{pred['lower'][0]:.2f}, {pred['upper'][0]:.2f}]")
        
        elif choice == '4':
            if modelo.is_trained:
                path = input("Caminho para salvar (Enter para padrão): ").strip()
                modelo.save_model(path if path else 'models/modelo_climatico.pkl')
            else:
                print("❌ Treine o modelo primeiro!")
        
        elif choice == '5':
            path = input("Caminho do modelo (Enter para padrão): ").strip()
            try:
                modelo.load_model(path if path else 'models/modelo_climatico.pkl')
                print("✅ Modelo carregado!")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif choice == '6':
            print(modelo.generate_report())
        
        elif choice == '0':
            print("👋 Encerrando...")
            break

if __name__ == "__main__":
    main()