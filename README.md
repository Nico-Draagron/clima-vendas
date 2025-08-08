# 🌤️ Sistema de Previsão Climática e Vendas

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
