# ============================================================================
# 📊 pages/dashboard_preditivo.py - DASHBOARD PREDITIVO INTEGRADO
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar modelo preditivo
try:
    from modelo_preditivo import ModeloVendasBootstrap
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

class DashboardPreditivo:
    """Dashboard preditivo integrado com widgets inteligentes"""
    
    def __init__(self, store_manager, auth_manager):
        self.store_manager = store_manager
        self.auth_manager = auth_manager
    
    def render_prediction_widgets(self, selected_store_id):
        """Renderiza widgets preditivos no dashboard principal"""
        
        if not MODEL_AVAILABLE:
            st.info("ℹ️ Modelo preditivo não disponível. Widgets desabilitados.")
            return
        
        # Verificar se há modelo treinado na sessão
        if 'modelo_treinado' not in st.session_state:
            st.info("🤖 Treine um modelo na página 'Modelo Preditivo' para ver previsões aqui")
            return
        
        stores = self.store_manager.get_available_stores()
        if selected_store_id not in stores:
            return
        
        # Carregar dados da loja
        df = self.store_manager.load_store_data(selected_store_id)
        if df is None or df.empty:
            return
        
        store_info = stores[selected_store_id]
        value_col = store_info['value_column']
        
        # Renderizar widgets
        self._render_7day_forecast_widget(df, value_col, store_info['display_name'])
        self._render_alerts_widget(df, value_col)
        self._render_prediction_accuracy_widget(df, value_col)
        self._render_confidence_score_widget()
    
    def _render_7day_forecast_widget(self, df, value_col, store_name):
        """Widget de previsão para próximos 7 dias"""
        
        st.subheader("🔮 Previsão 7 Dias")
        
        try:
            modelo = st.session_state['modelo_treinado']
            features = st.session_state['features_modelo']
            
            # Preparar dados para previsão
            df = self._prepare_future_data(df, features)
            
            # Gerar previsão de 7 dias
            last_date = df['data'].max()
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=7, freq='D')
            
            future_data = []
            for date in forecast_dates:
                row_data = self._create_future_row(date, df, features)
                future_data.append(row_data)
            
            future_df = pd.DataFrame(future_data)
            
            # Fazer previsões
            results = modelo.prever(future_df[features], usar_ensemble=True, retornar_intervalo=True)
            
            # Preparar dados para visualização
            forecast_data = pd.DataFrame({
                'Data': forecast_dates,
                'Previsão': results['predicao'],
                'IC_Inferior': results['intervalo_inferior'],
                'IC_Superior': results['intervalo_superior']
            })
            
            # Métricas resumo
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_prediction = forecast_data['Previsão'].mean()
                st.metric("💰 Média Prevista", f"R$ {avg_prediction:,.2f}".replace(',', '.'))
            
            with col2:
                total_prediction = forecast_data['Previsão'].sum()
                st.metric("📊 Total Semanal", f"R$ {total_prediction:,.2f}".replace(',', '.'))
            
            with col3:
                # Comparar com média histórica
                historical_avg = df[value_col].tail(30).mean()  # Últimos 30 dias
                trend = "📈" if avg_prediction > historical_avg else "📉" if avg_prediction < historical_avg else "➡️"
                change_pct = ((avg_prediction - historical_avg) / historical_avg) * 100
                st.metric("🎯 vs Histórico", f"{change_pct:+.1f}%", trend)
            
            # Gráfico compacto de previsão
            fig = go.Figure()
            
            # Dados históricos recentes (últimos 14 dias)
            recent_data = df.tail(14)
            
            fig.add_trace(go.Scatter(
                x=recent_data['data'],
                y=recent_data[value_col],
                mode='lines+markers',
                name='Histórico',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Previsões
            fig.add_trace(go.Scatter(
                x=forecast_data['Data'],
                y=forecast_data['Previsão'],
                mode='lines+markers',
                name='Previsão',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Intervalo de confiança
            fig.add_trace(go.Scatter(
                x=forecast_data['Data'],
                y=forecast_data['IC_Superior'],
                mode='lines',
                name='IC Superior',
                line=dict(color='lightcoral', width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Data'],
                y=forecast_data['IC_Inferior'],
                mode='lines',
                name='IC Inferior',
                fill='tonexty',
                fillcolor='rgba(255, 182, 193, 0.3)',
                line=dict(color='lightcoral', width=1),
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"Previsão de Vendas - {store_name}",
                xaxis_title="Data",
                yaxis_title="Vendas (R$)",
                height=350,
                hovermode='x unified',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela resumida
            with st.expander("📋 Detalhes da Previsão"):
                display_df = forecast_data.copy()
                display_df['Data'] = display_df['Data'].dt.strftime('%d/%m (%a)')
                display_df['Previsão'] = display_df['Previsão'].apply(lambda x: f"R$ {x:,.2f}")
                display_df['Faixa'] = display_df.apply(
                    lambda x: f"R$ {x['IC_Inferior']:,.2f} - R$ {x['IC_Superior']:,.2f}", axis=1
                )
                
                st.dataframe(
                    display_df[['Data', 'Previsão', 'Faixa']],
                    use_container_width=True,
                    hide_index=True
                )
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar previsão: {str(e)}")
    
    def _render_alerts_widget(self, df, value_col):
        """Widget de alertas baseados em previsões"""
        
        st.subheader("🚨 Alertas Preditivos")
        
        try:
            modelo = st.session_state['modelo_treinado']
            features = st.session_state['features_modelo']
            
            # Gerar alertas baseados nas previsões
            alerts = self._generate_prediction_alerts(df, value_col, modelo, features)
            
            if alerts:
                for alert in alerts:
                    if alert['type'] == 'error':
                        st.error(f"🚨 **{alert['title']}**: {alert['message']}")
                    elif alert['type'] == 'warning':
                        st.warning(f"⚠️ **{alert['title']}**: {alert['message']}")
                    elif alert['type'] == 'success':
                        st.success(f"✅ **{alert['title']}**: {alert['message']}")
                    else:
                        st.info(f"ℹ️ **{alert['title']}**: {alert['message']}")
            else:
                st.success("✅ **Sem Alertas**: Previsões dentro do esperado")
        
        except Exception as e:
            st.error(f"❌ Erro ao gerar alertas: {str(e)}")
    
    def _render_prediction_accuracy_widget(self, df, value_col):
        """Widget de comparação predito vs real"""
        
        st.subheader("🎯 Acurácia do Modelo")
        
        try:
            modelo = st.session_state['modelo_treinado']
            features = st.session_state['features_modelo']
            
            # Usar últimos 30 dias para validação
            validation_data = df.tail(30)
            
            if len(validation_data) < 5:
                st.info("ℹ️ Dados insuficientes para validação de acurácia")
                return
            
            # Preparar dados
            features_df = validation_data[features].fillna(validation_data[features].mean())
            target_df = validation_data[value_col]
            
            # Fazer previsões
            predictions = modelo.prever(features_df, usar_ensemble=True, retornar_intervalo=False)
            
            # Calcular métricas
            mae = np.mean(np.abs(target_df - predictions['predicao']))
            mape = np.mean(np.abs((target_df - predictions['predicao']) / target_df)) * 100
            accuracy = 100 - mape
            
            # Exibir métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 MAE", f"R$ {mae:,.2f}".replace(',', '.'))
            
            with col2:
                st.metric("📈 MAPE", f"{mape:.1f}%")
            
            with col3:
                color = "normal" if accuracy > 80 else "inverse"
                st.metric("🎯 Acurácia", f"{accuracy:.1f}%", delta_color=color)
            
            # Gráfico compacto de comparação
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=validation_data['data'],
                y=target_df,
                mode='lines+markers',
                name='Real',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=validation_data['data'],
                y=predictions['predicao'],
                mode='lines+markers',
                name='Previsto',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Real vs Previsto (Últimos 30 dias)",
                xaxis_title="Data",
                yaxis_title="Vendas (R$)",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erro ao calcular acurácia: {str(e)}")
    
    def _render_confidence_score_widget(self):
        """Widget de score de confiabilidade do modelo"""
        
        st.subheader("📊 Confiabilidade do Modelo")
        
        try:
            relatorio = st.session_state['relatorio_modelo']
            
            # Extrair métricas de confiabilidade
            r2_mean = relatorio['metricas']['R²']['media']
            r2_std = relatorio['metricas']['R²']['std']
            rmse_mean = relatorio['metricas']['RMSE']['media']
            
            # Calcular score de confiabilidade (0-100)
            # Baseado em R² (0-70%) + Estabilidade (0-30%)
            r2_score = max(0, min(70, r2_mean * 70))
            stability_score = max(0, min(30, (1 - r2_std) * 30))
            confidence_score = r2_score + stability_score
            
            # Determinar nível de confiança
            if confidence_score >= 80:
                confidence_level = "Alta"
                confidence_color = "🟢"
            elif confidence_score >= 60:
                confidence_level = "Média"
                confidence_color = "🟡"
            else:
                confidence_level = "Baixa"
                confidence_color = "🔴"
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Score Confiabilidade", f"{confidence_score:.0f}/100")
            
            with col2:
                st.metric("📈 Nível", f"{confidence_color} {confidence_level}")
            
            with col3:
                st.metric("📊 R² Médio", f"{r2_mean:.3f}")
            
            # Gauge de confiabilidade
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score de Confiabilidade"},
                delta={'reference': 80, 'position': "top"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Interpretação do score
            interpretations = {
                "Alta": "✅ Modelo altamente confiável. Previsões são precisas e estáveis.",
                "Média": "⚠️ Modelo moderadamente confiável. Use com cautela em decisões críticas.",
                "Baixa": "❌ Modelo pouco confiável. Retreine com mais dados ou ajuste parâmetros."
            }
            
            st.info(f"**Interpretação**: {interpretations[confidence_level]}")
            
        except Exception as e:
            st.error(f"❌ Erro ao calcular confiabilidade: {str(e)}")
    
    def _prepare_future_data(self, df, features):
        """Prepara dados para previsão futura"""
        
        df = df.copy()
        df['data'] = pd.to_datetime(df['data'])
        
        # Adicionar features temporais se não existirem
        if 'mes' not in df.columns:
            df['mes'] = df['data'].dt.month
        if 'dia_semana' not in df.columns:
            df['dia_semana'] = df['data'].dt.dayofweek
        if 'dia_mes' not in df.columns:
            df['dia_mes'] = df['data'].dt.day
        if 'dia_ano' not in df.columns:
            df['dia_ano'] = df['data'].dt.dayofyear
        if 'eh_weekend' not in df.columns:
            df['eh_weekend'] = df['dia_semana'].isin([5, 6]).astype(int)
        if 'trimestre' not in df.columns:
            df['trimestre'] = df['data'].dt.quarter
        
        # Features cíclicas
        if 'mes_sin' not in df.columns:
            df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        if 'mes_cos' not in df.columns:
            df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        if 'dia_ano_sin' not in df.columns:
            df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365)
        if 'dia_ano_cos' not in df.columns:
            df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365)
        
        return df
    
    def _create_future_row(self, date, df, features):
        """Cria linha de dados futuros"""
        
        row_data = {'data': date}
        
        # Features temporais
        row_data['mes'] = date.month
        row_data['dia_semana'] = date.dayofweek
        row_data['dia_mes'] = date.day
        row_data['dia_ano'] = date.dayofyear
        row_data['eh_weekend'] = 1 if date.dayofweek >= 5 else 0
        row_data['trimestre'] = (date.month - 1) // 3 + 1
        
        # Features cíclicas
        row_data['mes_sin'] = np.sin(2 * np.pi * date.month / 12)
        row_data['mes_cos'] = np.cos(2 * np.pi * date.month / 12)
        row_data['dia_ano_sin'] = np.sin(2 * np.pi * date.dayofyear / 365)
        row_data['dia_ano_cos'] = np.cos(2 * np.pi * date.dayofyear / 365)
        
        # Features climáticas (usar médias históricas)
        climate_features = [f for f in features if any(x in f for x in ['temp', 'precip', 'umid', 'rad'])]
        
        for feature in climate_features:
            if feature in df.columns:
                row_data[feature] = df[feature].mean()
            else:
                row_data[feature] = 0
        
        # Preencher features faltantes
        for feature in features:
            if feature not in row_data:
                if feature in df.columns:
                    row_data[feature] = df[feature].mean()
                else:
                    row_data[feature] = 0
        
        return row_data
    
    def _generate_prediction_alerts(self, df, value_col, modelo, features):
        """Gera alertas baseados nas previsões"""
        
        alerts = []
        
        try:
            # Previsão para amanhã
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_data = self._create_future_row(tomorrow, self._prepare_future_data(df, features), features)
            tomorrow_df = pd.DataFrame([tomorrow_data])
            
            tomorrow_prediction = modelo.prever(tomorrow_df[features], usar_ensemble=True, retornar_intervalo=True)
            predicted_sales = tomorrow_prediction['predicao'][0]
            
            # Comparar com médias históricas
            recent_avg = df[value_col].tail(30).mean()
            historical_avg = df[value_col].mean()
            
            # Alertas de tendência
            if predicted_sales < recent_avg * 0.8:
                alerts.append({
                    'type': 'warning',
                    'title': 'Queda Prevista nas Vendas',
                    'message': f'Vendas de amanhã ({predicted_sales:.0f}) 20%+ abaixo da média recente ({recent_avg:.0f})'
                })
            elif predicted_sales > recent_avg * 1.2:
                alerts.append({
                    'type': 'success',
                    'title': 'Alta nas Vendas Prevista',
                    'message': f'Vendas de amanhã ({predicted_sales:.0f}) 20%+ acima da média recente ({recent_avg:.0f})'
                })
            
            # Previsão de 7 dias
            future_dates = pd.date_range(datetime.now() + timedelta(days=1), periods=7, freq='D')
            future_data = [self._create_future_row(date, self._prepare_future_data(df, features), features) for date in future_dates]
            future_df = pd.DataFrame(future_data)
            
            week_predictions = modelo.prever(future_df[features], usar_ensemble=True, retornar_intervalo=True)
            week_total = week_predictions['predicao'].sum()
            week_avg = week_predictions['predicao'].mean()
            
            # Alertas semanais
            if week_avg < historical_avg * 0.85:
                alerts.append({
                    'type': 'warning',
                    'title': 'Semana Fraca Prevista',
                    'message': f'Média semanal prevista ({week_avg:.0f}) significativamente abaixo do histórico'
                })
            elif week_avg > historical_avg * 1.15:
                alerts.append({
                    'type': 'success',
                    'title': 'Semana Forte Prevista',
                    'message': f'Média semanal prevista ({week_avg:.0f}) significativamente acima do histórico'
                })
            
            # Alertas de volatilidade
            week_volatility = np.std(week_predictions['predicao'])
            historical_volatility = np.std(df[value_col].tail(30))
            
            if week_volatility > historical_volatility * 1.5:
                alerts.append({
                    'type': 'info',
                    'title': 'Alta Volatilidade Prevista',
                    'message': f'Vendas da próxima semana podem ter maior variação que o normal'
                })
            
            # Alertas de confiança
            relatorio = st.session_state['relatorio_modelo']
            r2_mean = relatorio['metricas']['R²']['media']
            
            if r2_mean < 0.6:
                alerts.append({
                    'type': 'warning',
                    'title': 'Baixa Confiabilidade do Modelo',
                    'message': f'Modelo tem R²={r2_mean:.3f}. Use previsões com cautela.'
                })
        
        except Exception as e:
            alerts.append({
                'type': 'error',
                'title': 'Erro ao Gerar Alertas',
                'message': f'Erro no sistema de alertas: {str(e)}'
            })
        
        return alerts

# ============================================================================
# 📊 INTEGRAÇÃO COM DASHBOARD PRINCIPAL
# ============================================================================

def add_prediction_widgets_to_dashboard(store_manager, auth_manager, selected_store_id):
    """
    Função para adicionar widgets preditivos ao dashboard principal.
    
    Esta função deve ser chamada dentro do dashboard principal (show_dashboard_page)
    """
    
    # Verificar permissões
    if not auth_manager.has_permission('use_models'):
        return
    
    # Criar instância do dashboard preditivo
    dashboard_pred = DashboardPreditivo(store_manager, auth_manager)
    
    # Renderizar seção preditiva
    st.markdown("---")
    st.markdown("## 🤖 Inteligência Preditiva")
    
    # Renderizar widgets
    dashboard_pred.render_prediction_widgets(selected_store_id)

# ============================================================================
# 📊 EXEMPLO DE INTEGRAÇÃO NO DASHBOARD PRINCIPAL
# ============================================================================

def enhanced_dashboard_with_predictions(df, role, store_manager, auth_manager, selected_store_id):
    """
    Versão melhorada do dashboard principal com widgets preditivos integrados.
    
    Esta função substitui ou complementa o show_dashboard_page existente.
    """
    
    st.header("📊 Dashboard Principal")
    
    if df is None:
        st.warning("⚠️ Selecione um dataset no menu lateral")
        return
    
    value_col = 'valor_loja_01'  # Ajustar conforme necessário
    
    # Métricas principais (existentes)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if value_col in df.columns:
            total_vendas = df[value_col].sum()
            st.metric("💰 Total Vendas", f"R$ {total_vendas:,.2f}".replace(',', '.'))
        else:
            st.metric("💰 Total Vendas", "N/A")
    
    with col2:
        st.metric("📅 Período", f"{len(df)} dias")
    
    with col3:
        if value_col in df.columns:
            media_vendas = df[value_col].mean()
            st.metric("📊 Média Diária", f"R$ {media_vendas:,.2f}".replace(',', '.'))
        else:
            st.metric("📊 Média Diária", "N/A")
    
    with col4:
        if 'precipitacao_total' in df.columns:
            dias_chuva = (df['precipitacao_total'] > 0).sum()
            st.metric("🌧️ Dias com Chuva", f"{dias_chuva}")
        else:
            st.metric("🌧️ Dias com Chuva", "N/A")
    
    # Gráfico de vendas por tempo (existente)
    if 'data' in df.columns and value_col in df.columns:
        st.subheader("📈 Evolução das Vendas")
        
        df_plot = df.copy()
        df_plot['data'] = pd.to_datetime(df_plot['data'])
        
        fig = px.line(
            df_plot, 
            x='data', 
            y=value_col,
            title="Faturamento Diário ao Longo do Tempo"
        )
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # **NOVA SEÇÃO**: Widgets Preditivos Integrados
    add_prediction_widgets_to_dashboard(store_manager, auth_manager, selected_store_id)
    
    # Dados protegidos por role (existentes)
    if role == "admin":
        st.subheader("🔒 Informações Confidenciais (Admin Only)")
        st.info("Esta seção só é visível para administradores")
        
        if value_col in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Maior Venda", f"R$ {df[value_col].max():,.2f}")
            with col2:
                st.metric("📉 Menor Venda", f"R$ {df[value_col].min():,.2f}")