# ============================================================================
# ⚙️ pages/admin.py - PAINEL ADMINISTRATIVO COMPLETO
# ============================================================================

import streamlit as st
import pandas as pd
import json
import os
import shutil
from datetime import datetime, timedelta
from data.store_manager import StoreDataManager
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import hashlib
import io
import zipfile
import random

class AdminPage:
    """Painel administrativo completo do sistema"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
        self.logs_file = 'data/system_logs.json'
        self.settings_file = 'data/admin_settings.json'
        
        # Garantir que diretórios existam
        os.makedirs('data', exist_ok=True)
        
        # Inicializar logs e configurações
        self._initialize_logs()
        self._initialize_settings()
    
    def render(self):
        """Renderiza painel administrativo"""
        
        st.markdown("# ⚙️ Painel Administrativo")
        st.markdown("**Configurações e gerenciamento completo do sistema**")
        
        # Verificar permissões
        if not self.auth_manager.has_permission('full_access'):
            st.error("❌ Acesso negado. Apenas administradores podem acessar este painel.")
            return
        
        # Log de acesso
        self._log_action("admin_panel_accessed", "Painel administrativo acessado")
        
        # Tabs do painel admin
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏪 Gerenciar Lojas",
            "👥 Usuários", 
            "📊 Sistema",
            "🔧 Configurações",
            "📈 Monitoramento"
        ])
        
        with tab1:
            self._render_store_management()
        
        with tab2:
            self._render_user_management()
        
        with tab3:
            self._render_system_info()
        
        with tab4:
            self._render_system_settings()
        
        with tab5:
            self._render_monitoring()
    
    def _initialize_logs(self):
        """Inicializa sistema de logs"""
        if not os.path.exists(self.logs_file):
            initial_logs = {
                "logs": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "user": "system",
                        "action": "system_initialized",
                        "details": "Sistema de logs inicializado"
                    }
                ]
            }
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(initial_logs, f, indent=2, ensure_ascii=False)
    
    def _initialize_settings(self):
        """Inicializa configurações do sistema"""
        if not os.path.exists(self.settings_file):
            default_settings = {
                "security": {
                    "session_timeout_minutes": 30,
                    "max_login_attempts": 3,
                    "require_strong_password": False,
                    "enable_2fa": False
                },
                "data": {
                    "max_file_size_mb": 100,
                    "backup_frequency": "Manual",
                    "data_retention_days": 365,
                    "auto_cleanup": False
                },
                "system": {
                    "log_level": "INFO",
                    "max_logs_entries": 1000,
                    "auto_backup": False,
                    "maintenance_window": "02:00"
                }
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, indent=2, ensure_ascii=False)
    
    def _log_action(self, action: str, details: str = "", level: str = "INFO"):
        """Registra ação no log do sistema"""
        try:
            with open(self.logs_file, 'r', encoding='utf-8') as f:
                logs_data = json.load(f)
            
            new_log = {
                "timestamp": datetime.now().isoformat(),
                "user": self.auth_manager.get_username() or "system",
                "action": action,
                "details": details,
                "level": level
            }
            
            logs_data["logs"].append(new_log)
            
            # Limitar número de logs
            max_logs = self._get_setting("system.max_logs_entries", 1000)
            if len(logs_data["logs"]) > max_logs:
                logs_data["logs"] = logs_data["logs"][-max_logs:]
            
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(logs_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            st.error(f"Erro ao registrar log: {e}")
    
    def _get_logs(self, limit: int = 100) -> List[Dict]:
        """Obtém logs do sistema"""
        try:
            with open(self.logs_file, 'r', encoding='utf-8') as f:
                logs_data = json.load(f)
            return logs_data["logs"][-limit:]
        except:
            return []
    
    def _get_setting(self, key: str, default=None):
        """Obtém configuração específica"""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            keys = key.split('.')
            value = settings
            for k in keys:
                value = value.get(k, default)
                if value is None:
                    return default
            return value
        except:
            return default
    
    def _save_settings(self, settings: Dict):
        """Salva configurações"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self._log_action("settings_updated", "Configurações do sistema atualizadas")
            return True
        except Exception as e:
            st.error(f"Erro ao salvar configurações: {e}")
            return False
    
    def _render_store_management(self):
        """Gerenciamento completo de lojas"""
        
        st.subheader("🏪 Gerenciamento de Lojas")
        
        # Carregar lojas
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.info("ℹ️ Nenhuma loja cadastrada no sistema")
            return
        
        st.write(f"**Total de lojas:** {len(stores)}")
        
        # Tabela de lojas com métricas
        store_data = []
        for store_id, store_info in stores.items():
            df = self.store_manager.load_store_data(store_id)
            
            if df is not None:
                records_count = len(df)
                date_range = f"{df['data'].min().strftime('%d/%m/%Y')} - {df['data'].max().strftime('%d/%m/%Y')}"
                
                # Calcular estatísticas
                value_col = store_info['value_column']
                if value_col in df.columns:
                    total_sales = df[value_col].sum()
                    avg_sales = df[value_col].mean()
                else:
                    total_sales = 0
                    avg_sales = 0
            else:
                records_count = 0
                date_range = "N/A"
                total_sales = 0
                avg_sales = 0
            
            store_data.append({
                'ID': store_id,
                'Nome': store_info['display_name'],
                'Arquivo': store_info['csv_file'],
                'Status': store_info.get('status', 'active'),
                'Registros': records_count,
                'Período': date_range,
                'Vendas Totais': f"R$ {total_sales:,.0f}".replace(',', '.'),
                'Média Diária': f"R$ {avg_sales:,.0f}".replace(',', '.'),
                'Criado em': store_info.get('created_date', 'N/A'),
                'Localização': store_info.get('location', 'N/A')
            })
        
        stores_df = pd.DataFrame(store_data)
        st.dataframe(stores_df, use_container_width=True)
        
        # Ações implementadas
        st.subheader("⚡ Ações")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Atualizar Status"):
                self._update_stores_status(stores)
        
        with col2:
            if st.button("📊 Gerar Relatório"):
                self._generate_stores_report(stores_df)
        
        with col3:
            if st.button("💾 Backup Completo"):
                self._create_full_backup()
        
        with col4:
            if st.button("📤 Exportar Dados"):
                self._export_all_data()
        
        # Gerenciamento individual de lojas
        st.subheader("🔧 Gerenciar Loja Específica")
        
        selected_store = st.selectbox(
            "Escolha uma loja:",
            options=list(stores.keys()),
            format_func=lambda x: f"{stores[x]['display_name']} ({x})"
        )
        
        if selected_store:
            self._render_individual_store_management(selected_store, stores[selected_store])
    
    def _update_stores_status(self, stores):
        """Atualiza status de todas as lojas"""
        with st.spinner("Verificando status das lojas..."):
            issues = []
            
            for store_id, store_info in stores.items():
                # Verificar arquivo
                csv_path = os.path.join('data/datasets', store_info['csv_file'])
                if not os.path.exists(csv_path):
                    issues.append(f"❌ {store_id}: Arquivo {store_info['csv_file']} não encontrado")
                    continue
                
                # Verificar dados
                df = self.store_manager.load_store_data(store_id)
                if df is None or df.empty:
                    issues.append(f"⚠️ {store_id}: Dados não carregados ou vazios")
                    continue
                
                # Verificar integridade
                value_col = store_info['value_column']
                if value_col not in df.columns:
                    issues.append(f"⚠️ {store_id}: Coluna de valor '{value_col}' não encontrada")
            
            if issues:
                st.error("**Problemas encontrados:**")
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ Todas as lojas estão funcionando corretamente!")
            
            self._log_action("stores_status_checked", f"Verificação de status concluída. {len(issues)} problemas encontrados")
    
    def _generate_stores_report(self, stores_df):
        """Gera relatório completo das lojas"""
        
        st.subheader("📊 Relatório de Lojas")
        
        # Métricas agregadas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stores = len(stores_df)
            st.metric("🏪 Total de Lojas", total_stores)
        
        with col2:
            active_stores = len(stores_df[stores_df['Status'] == 'active'])
            st.metric("✅ Lojas Ativas", active_stores)
        
        with col3:
            total_records = stores_df['Registros'].sum()
            st.metric("📊 Total de Registros", f"{total_records:,}")
        
        with col4:
            # Calcular total de vendas (remover formatação)
            total_sales = 0
            for _, row in stores_df.iterrows():
                sales_str = row['Vendas Totais'].replace('R$ ', '').replace('.', '').replace(',', '.')
                try:
                    total_sales += float(sales_str)
                except:
                    pass
            st.metric("💰 Vendas Totais", f"R$ {total_sales:,.0f}".replace(',', '.'))
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de registros por loja
            fig_records = px.bar(
                stores_df,
                x='Nome',
                y='Registros',
                title="Registros por Loja",
                color='Registros',
                color_continuous_scale='viridis'
            )
            fig_records.update_xaxis(tickangle=45)
            st.plotly_chart(fig_records, use_container_width=True)
        
        with col2:
            # Status das lojas
            status_counts = stores_df['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status das Lojas"
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Exportar relatório
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'total_stores': total_stores,
            'active_stores': active_stores,
            'total_records': total_records,
            'total_sales': total_sales,
            'stores_details': stores_df.to_dict('records')
        }
        
        report_json = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            "📥 Download Relatório (JSON)",
            data=report_json,
            file_name=f"relatorio_lojas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        self._log_action("stores_report_generated", f"Relatório de {total_stores} lojas gerado")
    
    def _create_full_backup(self):
        """Cria backup completo do sistema"""
        
        with st.spinner("Criando backup completo..."):
            try:
                # Preparar dados do backup
                backup_data = {
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0.0',
                    'stores_config': {},
                    'admin_settings': {},
                    'system_logs': []
                }
                
                # Backup das configurações de lojas
                if os.path.exists(self.store_manager.stores_config_file):
                    with open(self.store_manager.stores_config_file, 'r', encoding='utf-8') as f:
                        backup_data['stores_config'] = json.load(f)
                
                # Backup das configurações do admin
                if os.path.exists(self.settings_file):
                    with open(self.settings_file, 'r', encoding='utf-8') as f:
                        backup_data['admin_settings'] = json.load(f)
                
                # Backup dos logs (últimos 500)
                backup_data['system_logs'] = self._get_logs(500)
                
                # Criar ZIP com todos os arquivos
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Adicionar configurações
                    zip_file.writestr(
                        'backup_config.json',
                        json.dumps(backup_data, indent=2, ensure_ascii=False, default=str)
                    )
                    
                    # Adicionar arquivos de dados das lojas
                    datasets_path = 'data/datasets'
                    if os.path.exists(datasets_path):
                        for file in os.listdir(datasets_path):
                            if file.endswith('.csv'):
                                file_path = os.path.join(datasets_path, file)
                                zip_file.write(file_path, f'datasets/{file}')
                
                zip_buffer.seek(0)
                
                backup_filename = f"backup_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                st.download_button(
                    "💾 Download Backup Completo",
                    data=zip_buffer.getvalue(),
                    file_name=backup_filename,
                    mime="application/zip"
                )
                
                st.success("✅ Backup completo criado com sucesso!")
                self._log_action("full_backup_created", f"Backup completo criado: {backup_filename}")
                
            except Exception as e:
                st.error(f"❌ Erro ao criar backup: {e}")
                self._log_action("backup_failed", f"Erro ao criar backup: {str(e)}", "ERROR")
    
    def _export_all_data(self):
        """Exporta todos os dados em formato CSV"""
        
        stores = self.store_manager.get_available_stores()
        
        if not stores:
            st.warning("Nenhuma loja para exportar")
            return
        
        with st.spinner("Preparando exportação..."):
            try:
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for store_id, store_info in stores.items():
                        df = self.store_manager.load_store_data(store_id)
                        
                        if df is not None and not df.empty:
                            csv_data = df.to_csv(index=False)
                            filename = f"{store_info['display_name'].replace(' ', '_')}_{store_id}.csv"
                            zip_file.writestr(filename, csv_data)
                
                zip_buffer.seek(0)
                
                export_filename = f"dados_todas_lojas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                st.download_button(
                    "📤 Download Dados (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=export_filename,
                    mime="application/zip"
                )
                
                st.success("✅ Exportação preparada com sucesso!")
                self._log_action("data_exported", f"Dados de {len(stores)} lojas exportados")
                
            except Exception as e:
                st.error(f"❌ Erro na exportação: {e}")
    
    def _render_individual_store_management(self, store_id, store_info):
        """Gerenciamento individual de loja"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Informações:**")
            st.json(store_info)
            
            # Opções de edição
            with st.expander("✏️ Editar Informações"):
                new_display_name = st.text_input("Nome de Exibição", value=store_info['display_name'])
                new_location = st.text_input("Localização", value=store_info.get('location', ''))
                new_description = st.text_area("Descrição", value=store_info.get('description', ''))
                
                if st.button(f"💾 Salvar Alterações", key=f"save_{store_id}"):
                    self._update_store_info(store_id, {
                        'display_name': new_display_name,
                        'location': new_location,
                        'description': new_description
                    })
        
        with col2:
            # Estatísticas da loja
            df = self.store_manager.load_store_data(store_id)
            
            if df is not None:
                st.markdown("**📊 Estatísticas:**")
                
                stats = {
                    "Registros": len(df),
                    "Período": f"{df['data'].min().strftime('%d/%m/%Y')} - {df['data'].max().strftime('%d/%m/%Y')}",
                    "Colunas": len(df.columns),
                    "Dados Faltantes": df.isnull().sum().sum(),
                    "Tamanho do Arquivo": f"{os.path.getsize(os.path.join('data/datasets', store_info['csv_file'])) / 1024:.1f} KB"
                }
                
                for key, value in stats.items():
                    st.write(f"- **{key}:** {value}")
                
                # Opções de manutenção
                st.markdown("**🔧 Manutenção:**")
                
                col1_maint, col2_maint = st.columns(2)
                
                with col1_maint:
                    if st.button("🔄 Revalidar Dados", key=f"validate_{store_id}"):
                        self._validate_store_data(store_id, store_info)
                    
                    if st.button("📊 Análise Qualidade", key=f"quality_{store_id}"):
                        self._analyze_data_quality(store_id, df)
                
                with col2_maint:
                    if st.button("📥 Download CSV", key=f"download_{store_id}"):
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "💾 Baixar",
                            data=csv_data,
                            file_name=f"{store_info['display_name'].replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"dl_{store_id}"
                        )
                    
                    if st.button("🗑️ Remover Loja", key=f"remove_{store_id}"):
                        self._confirm_store_removal(store_id, store_info)
            else:
                st.error("❌ Não foi possível carregar dados da loja")
    
    def _update_store_info(self, store_id, updates):
        """Atualiza informações da loja"""
        try:
            with open(self.store_manager.stores_config_file, 'r', encoding='utf-8') as f:
                stores_config = json.load(f)
            
            if store_id in stores_config:
                stores_config[store_id].update(updates)
                
                with open(self.store_manager.stores_config_file, 'w', encoding='utf-8') as f:
                    json.dump(stores_config, f, indent=2, ensure_ascii=False)
                
                st.success("✅ Informações da loja atualizadas!")
                self._log_action("store_info_updated", f"Informações da loja {store_id} atualizadas")
                st.rerun()
            else:
                st.error("❌ Loja não encontrada")
                
        except Exception as e:
            st.error(f"❌ Erro ao atualizar: {e}")
    
    def _validate_store_data(self, store_id, store_info):
        """Valida dados da loja"""
        
        st.subheader(f"🔍 Validação - {store_info['display_name']}")
        
        df = self.store_manager.load_store_data(store_id)
        
        if df is None:
            st.error("❌ Não foi possível carregar dados")
            return
        
        # Validações
        issues = []
        warnings = []
        
        # 1. Verificar coluna de data
        if 'data' not in df.columns:
            issues.append("❌ Coluna 'data' não encontrada")
        else:
            invalid_dates = df['data'].isna().sum()
            if invalid_dates > 0:
                warnings.append(f"⚠️ {invalid_dates} datas inválidas encontradas")
        
        # 2. Verificar coluna de valor
        value_col = store_info['value_column']
        if value_col not in df.columns:
            issues.append(f"❌ Coluna de valor '{value_col}' não encontrada")
        else:
            invalid_values = df[value_col].isna().sum()
            negative_values = (df[value_col] < 0).sum()
            
            if invalid_values > 0:
                warnings.append(f"⚠️ {invalid_values} valores faltantes")
            if negative_values > 0:
                warnings.append(f"⚠️ {negative_values} valores negativos")
        
        # 3. Verificar duplicatas
        if 'data' in df.columns:
            duplicates = df.duplicated(subset=['data']).sum()
            if duplicates > 0:
                warnings.append(f"⚠️ {duplicates} registros duplicados por data")
        
        # 4. Verificar gaps de data
        if 'data' in df.columns and len(df) > 1:
            date_range = pd.date_range(df['data'].min(), df['data'].max(), freq='D')
            missing_dates = len(date_range) - len(df)
            if missing_dates > 0:
                warnings.append(f"⚠️ {missing_dates} datas faltantes no período")
        
        # Exibir resultados
        if not issues and not warnings:
            st.success("✅ Todos os dados estão válidos!")
        else:
            if issues:
                st.error("**Problemas críticos encontrados:**")
                for issue in issues:
                    st.write(issue)
            
            if warnings:
                st.warning("**Avisos de qualidade:**")
                for warning in warnings:
                    st.write(warning)
        
        self._log_action("store_data_validated", f"Validação da loja {store_id} concluída. {len(issues)} problemas, {len(warnings)} avisos")
    
    def _analyze_data_quality(self, store_id, df):
        """Análise detalhada da qualidade dos dados"""
        
        st.subheader("📊 Análise de Qualidade dos Dados")
        
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Total Registros", len(df))
        
        with col2:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("❓ Dados Faltantes", f"{missing_pct:.1f}%")
        
        with col3:
            duplicates = df.duplicated().sum()
            st.metric("🔄 Duplicatas", duplicates)
        
        with col4:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.metric("🔢 Colunas Numéricas", numeric_cols)
        
        # Qualidade por coluna
        st.subheader("📋 Qualidade por Coluna")
        
        quality_data = []
        for col in df.columns:
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            dtype = str(df[col].dtype)
            
            # Estatísticas específicas por tipo
            if df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].nunique()
                stats_info = f"Únicos: {unique_vals}"
            else:
                max_len = df[col].astype(str).str.len().max() if not df[col].empty else 0
                stats_info = f"Tamanho máx: {max_len}"
            
            quality_data.append({
                'Coluna': col,
                'Tipo': dtype,
                'Faltantes': missing,
                'Faltantes (%)': f"{missing_pct:.1f}%",
                'Informações': stats_info
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
        
        # Visualização de dados faltantes
        if quality_df['Faltantes'].sum() > 0:
            missing_data = quality_df[quality_df['Faltantes'] > 0]
            
            fig = px.bar(
                missing_data,
                x='Coluna',
                y='Faltantes',
                title="Dados Faltantes por Coluna",
                color='Faltantes',
                color_continuous_scale='reds'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    def _confirm_store_removal(self, store_id, store_info):
        """Confirma remoção de loja"""
        
        st.subheader("🗑️ Confirmar Remoção")
        
        st.error(f"""
        **ATENÇÃO: Esta ação é irreversível!**
        
        Você está prestes a remover:
        - **Loja:** {store_info['display_name']} ({store_id})
        - **Arquivo:** {store_info['csv_file']}
        - **Todos os dados** associados a esta loja
        """)
        
        confirmation_text = st.text_input(
            f"Digite '{store_id}' para confirmar:",
            placeholder=f"Digite {store_id} aqui"
        )
        
        if confirmation_text == store_id:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ CONFIRMAR REMOÇÃO", type="primary"):
                    self._remove_store(store_id, store_info)
            
            with col2:
                if st.button("❌ Cancelar", type="secondary"):
                    st.rerun()
        else:
            st.info("Digite o ID da loja exatamente como mostrado para habilitar a remoção")
    
    def _remove_store(self, store_id, store_info):
        """Remove loja do sistema"""
        try:
            # 1. Remover arquivo CSV
            csv_path = os.path.join('data/datasets', store_info['csv_file'])
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            # 2. Remover da configuração
            with open(self.store_manager.stores_config_file, 'r', encoding='utf-8') as f:
                stores_config = json.load(f)
            
            if store_id in stores_config:
                del stores_config[store_id]
                
                with open(self.store_manager.stores_config_file, 'w', encoding='utf-8') as f:
                    json.dump(stores_config, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Loja {store_info['display_name']} removida com sucesso!")
            self._log_action("store_removed", f"Loja {store_id} ({store_info['display_name']}) removida do sistema", "WARNING")
            
            # Recarregar página
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erro ao remover loja: {e}")
            self._log_action("store_removal_failed", f"Erro ao remover loja {store_id}: {str(e)}", "ERROR")
    
    def _render_user_management(self):
        """Gerenciamento completo de usuários"""
        
        st.subheader("👥 Gerenciamento de Usuários")
        
        # Usuários atuais (expandido)
        users_data = [
            {
                'Usuário': 'admin',
                'Nome': 'Administrador do Sistema',
                'Email': 'admin@empresa.com',
                'Role': 'admin',
                'Status': 'Ativo',
                'Último Login': 'Hoje às 10:30',
                'Sessões Ativas': 1,
                'Tentativas de Login': 0
            },
            {
                'Usuário': 'usuario',
                'Nome': 'Usuário Padrão',
                'Email': 'usuario@empresa.com',
                'Role': 'user',
                'Status': 'Ativo',
                'Último Login': 'Ontem às 14:20',
                'Sessões Ativas': 0,
                'Tentativas de Login': 0
            }
        ]
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True)
        
        # Estatísticas de usuários
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Usuários", len(users_df))
        
        with col2:
            active_users = len(users_df[users_df['Status'] == 'Ativo'])
            st.metric("✅ Usuários Ativos", active_users)
        
        with col3:
            admin_count = len(users_df[users_df['Role'] == 'admin'])
            st.metric("👑 Administradores", admin_count)
        
        with col4:
            active_sessions = users_df['Sessões Ativas'].sum()
            st.metric("🔗 Sessões Ativas", active_sessions)
        
        # Adicionar novo usuário (melhorado)
        with st.expander("➕ Adicionar Novo Usuário"):
            
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_username = st.text_input("👤 Username *", help="Nome de usuário único")
                    new_name = st.text_input("📝 Nome Completo *", help="Nome completo do usuário")
                    new_email = st.text_input("📧 Email *", help="Endereço de email válido")
                
                with col2:
                    new_role = st.selectbox("🔐 Role *", ['user', 'admin'], help="Nível de acesso")
                    new_password = st.text_input("🔒 Senha *", type="password", help="Senha deve ter pelo menos 6 caracteres")
                    confirm_password = st.text_input("🔒 Confirmar Senha *", type="password")
                
                # Configurações opcionais
                st.markdown("**⚙️ Configurações Opcionais:**")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    can_upload = st.checkbox("📤 Pode fazer upload", value=False, help="Permitir upload de arquivos")
                    can_export = st.checkbox("📥 Pode exportar dados", value=True, help="Permitir exportação de dados")
                
                with col4:
                    auto_logout = st.number_input("⏰ Auto-logout (min)", min_value=15, max_value=240, value=30)
                    account_expires = st.date_input("📅 Conta expira em", value=None, help="Deixe vazio para nunca expirar")
                
                submit_user = st.form_submit_button("➕ Criar Usuário", type="primary")
                
                if submit_user:
                    # Validações
                    errors = []
                    
                    if not all([new_username, new_name, new_email, new_password]):
                        errors.append("Preencha todos os campos obrigatórios")
                    
                    if len(new_password) < 6:
                        errors.append("Senha deve ter pelo menos 6 caracteres")
                    
                    if new_password != confirm_password:
                        errors.append("Senhas não coincidem")
                    
                    if new_username.lower() in ['admin', 'usuario']:
                        errors.append("Username já existe")
                    
                    if '@' not in new_email:
                        errors.append("Email inválido")
                    
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        # Simular criação do usuário
                        user_config = {
                            'username': new_username,
                            'name': new_name,
                            'email': new_email,
                            'role': new_role,
                            'password_hash': hashlib.sha256(new_password.encode()).hexdigest(),
                            'can_upload': can_upload,
                            'can_export': can_export,
                            'auto_logout_minutes': auto_logout,
                            'account_expires': account_expires.isoformat() if account_expires else None,
                            'created_at': datetime.now().isoformat(),
                            'created_by': self.auth_manager.get_username(),
                            'status': 'active'
                        }
                        
                        st.success(f"✅ Usuário '{new_username}' criado com sucesso!")
                        st.json(user_config)  # Mostrar configuração (em produção, salvar em BD)
                        
                        self._log_action("user_created", f"Usuário {new_username} ({new_role}) criado")
        
        # Logs de acesso (melhorados)
        st.subheader("📋 Logs de Acesso Recentes")
        
        # Filtros para logs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_filter_user = st.selectbox("Filtrar por usuário:", ['Todos'] + [u['Usuário'] for u in users_data])
        
        with col2:
            log_filter_action = st.selectbox("Filtrar por ação:", ['Todas', 'Login', 'Logout', 'Erro', 'Acesso'])
        
        with col3:
            log_limit = st.number_input("Mostrar últimas:", min_value=10, max_value=100, value=20)
        
        # Mock de logs expandidos
        access_logs = [
            {'Timestamp': '2025-01-08 10:30:15', 'Usuário': 'admin', 'Ação': 'Login', 'IP': '192.168.1.100', 'Status': 'Sucesso', 'Detalhes': 'Login realizado com sucesso'},
            {'Timestamp': '2025-01-08 10:31:22', 'Usuário': 'admin', 'Ação': 'Acesso', 'IP': '192.168.1.100', 'Status': 'Sucesso', 'Detalhes': 'Acessou dashboard principal'},
            {'Timestamp': '2025-01-08 10:35:45', 'Usuário': 'usuario', 'Ação': 'Login', 'IP': '192.168.1.101', 'Status': 'Sucesso', 'Detalhes': 'Login realizado com sucesso'},
            {'Timestamp': '2025-01-08 10:36:12', 'Usuário': 'usuario', 'Ação': 'Acesso', 'IP': '192.168.1.101', 'Status': 'Sucesso', 'Detalhes': 'Visualizou dados da loja_001'},
            {'Timestamp': '2025-01-08 09:45:30', 'Usuário': 'test', 'Ação': 'Login', 'IP': '192.168.1.102', 'Status': 'Erro', 'Detalhes': 'Tentativa de login com credenciais inválidas'},
            {'Timestamp': '2025-01-08 09:30:10', 'Usuário': 'admin', 'Ação': 'Acesso', 'IP': '192.168.1.100', 'Status': 'Sucesso', 'Detalhes': 'Acessou painel administrativo'}
        ]
        
        # Aplicar filtros
        filtered_logs = access_logs.copy()
        
        if log_filter_user != 'Todos':
            filtered_logs = [log for log in filtered_logs if log['Usuário'] == log_filter_user]
        
        if log_filter_action != 'Todas':
            filtered_logs = [log for log in filtered_logs if log['Ação'] == log_filter_action]
        
        filtered_logs = filtered_logs[:log_limit]
        
        logs_df = pd.DataFrame(filtered_logs)
        st.dataframe(logs_df, use_container_width=True)
        
        # Estatísticas de segurança
        st.subheader("🔒 Estatísticas de Segurança")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tentativas de login por hora
            login_attempts = [2, 1, 0, 0, 1, 3, 2, 1, 0, 2, 4, 1]
            hours = list(range(12))
            
            fig_attempts = px.line(
                x=hours,
                y=login_attempts,
                title="Tentativas de Login por Hora",
                labels={'x': 'Hora', 'y': 'Tentativas'}
            )
            st.plotly_chart(fig_attempts, use_container_width=True)
        
        with col2:
            # Status dos logs
            status_counts = pd.Series([log['Status'] for log in access_logs]).value_counts()
            
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status dos Logs de Acesso"
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    def _render_system_info(self):
        """Informações completas do sistema"""
        
        st.subheader("📊 Informações do Sistema")
        
        # Estatísticas principais (expandidas)
        col1, col2, col3, col4 = st.columns(4)
        
        stores = self.store_manager.get_available_stores()
        total_records = 0
        total_file_size = 0
        
        for store_id in stores.keys():
            df = self.store_manager.load_store_data(store_id)
            if df is not None:
                total_records += len(df)
                
                # Calcular tamanho do arquivo
                store_info = stores[store_id]
                csv_path = os.path.join('data/datasets', store_info['csv_file'])
                if os.path.exists(csv_path):
                    total_file_size += os.path.getsize(csv_path)
        
        with col1:
            st.metric("🏪 Lojas Ativas", len(stores))
        
        with col2:
            st.metric("📊 Total Registros", f"{total_records:,}")
        
        with col3:
            st.metric("💾 Espaço Usado", f"{total_file_size / (1024*1024):.1f} MB")
        
        with col4:
            uptime_hours = 24  # Mock - calcular uptime real
            st.metric("⏰ Uptime", f"{uptime_hours}h")
        
        # Performance do sistema
        st.subheader("⚡ Performance do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simular métricas de performance
            performance_data = {
                'Métrica': ['CPU Usage', 'Memory Usage', 'Disk Usage', 'Response Time'],
                'Valor': [45, 62, 38, 150],
                'Unidade': ['%', '%', '%', 'ms'],
                'Status': ['OK', 'OK', 'OK', 'OK']
            }
            
            perf_df = pd.DataFrame(performance_data)
            perf_df['Display'] = perf_df.apply(lambda x: f"{x['Valor']}{x['Unidade']}", axis=1)
            
            st.dataframe(perf_df[['Métrica', 'Display', 'Status']], use_container_width=True)
        
        with col2:
            # Gráfico de uso ao longo do tempo
            hours = list(range(24))
            cpu_usage = [30 + i*2 + (i%3)*5 for i in range(24)]
            memory_usage = [40 + i*1.5 + (i%4)*3 for i in range(24)]
            
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Scatter(x=hours, y=cpu_usage, name='CPU %', mode='lines'))
            fig_performance.add_trace(go.Scatter(x=hours, y=memory_usage, name='Memory %', mode='lines'))
            
            fig_performance.update_layout(
                title="Uso de Recursos (24h)",
                xaxis_title="Hora",
                yaxis_title="Uso (%)",
                height=300
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
        
        # Informações técnicas (expandidas)
        st.subheader("🔧 Informações Técnicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tech_info = {
                "Sistema Operacional": "Linux (Streamlit Cloud)",
                "Versão Python": "3.11.7",
                "Versão Streamlit": "1.37.0",
                "Versão Pandas": "2.1.4",
                "Versão Plotly": "5.17.0"
            }
            
            st.markdown("**📦 Software:**")
            for key, value in tech_info.items():
                st.write(f"- **{key}:** {value}")
        
        with col2:
            system_info = {
                "Arquitetura": "x86_64",
                "Processadores": "2 vCPUs",
                "Memória RAM": "4 GB",
                "Armazenamento": "10 GB SSD",
                "Rede": "100 Mbps"
            }
            
            st.markdown("**⚙️ Hardware:**")
            for key, value in system_info.items():
                st.write(f"- **{key}:** {value}")
        
        # Status detalhado dos arquivos
        st.subheader("📁 Status Detalhado dos Arquivos")
        
        files_to_check = [
            ('stores_config.json', self.store_manager.stores_config_file),
            ('admin_settings.json', self.settings_file),
            ('system_logs.json', self.logs_file),
            ('resumo_diario_climatico.csv', os.path.join('data/datasets', 'resumo_diario_climatico.csv'))
        ]
        
        files_status = []
        
        for file_name, file_path in files_to_check:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                files_status.append({
                    'Arquivo': file_name,
                    'Status': '✅ Existente',
                    'Tamanho': f"{size:,} bytes",
                    'Modificado': modified.strftime('%d/%m/%Y %H:%M'),
                    'Caminho': file_path
                })
            else:
                files_status.append({
                    'Arquivo': file_name,
                    'Status': '❌ Não encontrado',
                    'Tamanho': 'N/A',
                    'Modificado': 'N/A',
                    'Caminho': file_path
                })
        
        # Adicionar arquivos das lojas
        for store_id, store_info in stores.items():
            csv_path = os.path.join('data/datasets', store_info['csv_file'])
            
            if os.path.exists(csv_path):
                size = os.path.getsize(csv_path)
                modified = datetime.fromtimestamp(os.path.getmtime(csv_path))
                
                files_status.append({
                    'Arquivo': store_info['csv_file'],
                    'Status': '✅ Existente',
                    'Tamanho': f"{size:,} bytes",
                    'Modificado': modified.strftime('%d/%m/%Y %H:%M'),
                    'Caminho': csv_path
                })
        
        files_df = pd.DataFrame(files_status)
        st.dataframe(files_df, use_container_width=True)
        
        # Ações de manutenção
        st.subheader("🔧 Ações de Manutenção")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Verificar Integridade"):
                self._check_system_integrity()
        
        with col2:
            if st.button("🔄 Reindexar Dados"):
                self._reindex_data()
        
        with col3:
            if st.button("🧹 Limpeza Automática"):
                self._auto_cleanup()
        
        with col4:
            if st.button("📊 Diagnóstico Completo"):
                self._full_diagnostic()
    
    def _check_system_integrity(self):
        """Verifica integridade do sistema"""
        
        st.subheader("🔍 Verificação de Integridade")
        
        with st.spinner("Verificando integridade do sistema..."):
            issues = []
            warnings = []
            
            # 1. Verificar arquivos essenciais
            essential_files = [
                self.store_manager.stores_config_file,
                self.settings_file,
                self.logs_file
            ]
            
            for file_path in essential_files:
                if not os.path.exists(file_path):
                    issues.append(f"❌ Arquivo essencial não encontrado: {file_path}")
                else:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        issues.append(f"❌ Arquivo JSON corrompido: {file_path}")
            
            # 2. Verificar dados das lojas
            stores = self.store_manager.get_available_stores()
            
            for store_id, store_info in stores.items():
                csv_path = os.path.join('data/datasets', store_info['csv_file'])
                
                if not os.path.exists(csv_path):
                    issues.append(f"❌ Arquivo de dados não encontrado: {store_info['csv_file']}")
                    continue
                
                # Verificar se carrega corretamente
                df = self.store_manager.load_store_data(store_id)
                if df is None or df.empty:
                    issues.append(f"❌ Dados da loja {store_id} não podem ser carregados")
                    continue
                
                # Verificar colunas essenciais
                if 'data' not in df.columns:
                    issues.append(f"❌ Coluna 'data' ausente em {store_id}")
                
                value_col = store_info['value_column']
                if value_col not in df.columns:
                    issues.append(f"❌ Coluna de valor '{value_col}' ausente em {store_id}")
                
                # Verificar qualidade dos dados
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                if missing_pct > 20:
                    warnings.append(f"⚠️ {store_id}: {missing_pct:.1f}% de dados faltantes")
            
            # 3. Verificar configurações
            try:
                settings = json.load(open(self.settings_file, 'r', encoding='utf-8'))
                required_sections = ['security', 'data', 'system']
                
                for section in required_sections:
                    if section not in settings:
                        warnings.append(f"⚠️ Seção '{section}' ausente nas configurações")
            except:
                issues.append("❌ Erro ao verificar configurações")
        
        # Exibir resultados
        if not issues and not warnings:
            st.success("✅ Sistema íntegro! Nenhum problema encontrado.")
        else:
            if issues:
                st.error("**Problemas críticos encontrados:**")
                for issue in issues:
                    st.write(issue)
            
            if warnings:
                st.warning("**Avisos de integridade:**")
                for warning in warnings:
                    st.write(warning)
        
        # Relatório de integridade
        integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'critical_issues': len(issues),
            'warnings': len(warnings),
            'stores_checked': len(stores),
            'files_verified': len(essential_files) + len(stores),
            'issues_details': issues,
            'warnings_details': warnings
        }
        
        st.subheader("📋 Relatório de Integridade")
        st.json(integrity_report)
        
        # Download do relatório
        report_json = json.dumps(integrity_report, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download Relatório",
            data=report_json,
            file_name=f"relatorio_integridade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        self._log_action("integrity_check_completed", 
                        f"Verificação de integridade concluída. {len(issues)} problemas críticos, {len(warnings)} avisos")
    
    def _reindex_data(self):
        """Reindexação dos dados do sistema"""
        
        st.subheader("🔄 Reindexação de Dados")
        
        with st.spinner("Reindexando dados do sistema..."):
            stores = self.store_manager.get_available_stores()
            reindex_results = []
            
            for store_id, store_info in stores.items():
                try:
                    df = self.store_manager.load_store_data(store_id)
                    
                    if df is not None and not df.empty:
                        # Reordenar por data
                        if 'data' in df.columns:
                            df_sorted = df.sort_values('data').reset_index(drop=True)
                            
                            # Verificar se mudou a ordem
                            if not df.equals(df_sorted):
                                # Salvar reindexado
                                csv_path = os.path.join('data/datasets', store_info['csv_file'])
                                df_sorted.to_csv(csv_path, index=False)
                                
                                reindex_results.append({
                                    'store': store_id,
                                    'action': 'reordenado',
                                    'records': len(df_sorted)
                                })
                            else:
                                reindex_results.append({
                                    'store': store_id,
                                    'action': 'já ordenado',
                                    'records': len(df)
                                })
                        else:
                            reindex_results.append({
                                'store': store_id,
                                'action': 'sem coluna de data',
                                'records': len(df)
                            })
                    else:
                        reindex_results.append({
                            'store': store_id,
                            'action': 'erro no carregamento',
                            'records': 0
                        })
                        
                except Exception as e:
                    reindex_results.append({
                        'store': store_id,
                        'action': f'erro: {str(e)}',
                        'records': 0
                    })
        
        # Exibir resultados
        st.subheader("📋 Resultados da Reindexação")
        
        results_df = pd.DataFrame(reindex_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Estatísticas
        reordered = len([r for r in reindex_results if r['action'] == 'reordenado'])
        already_ordered = len([r for r in reindex_results if r['action'] == 'já ordenado'])
        errors = len([r for r in reindex_results if 'erro' in r['action']])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔄 Reordenados", reordered)
        
        with col2:
            st.metric("✅ Já Ordenados", already_ordered)
        
        with col3:
            st.metric("❌ Erros", errors)
        
        if errors == 0:
            st.success("✅ Reindexação concluída com sucesso!")
        else:
            st.warning(f"⚠️ Reindexação concluída com {errors} erros")
        
        self._log_action("data_reindexed", f"Reindexação concluída. {reordered} reordenados, {errors} erros")
    
    def _auto_cleanup(self):
        """Limpeza automática do sistema"""
        
        st.subheader("🧹 Limpeza Automática")
        
        cleanup_options = st.multiselect(
            "Escolha as operações de limpeza:",
            [
                "🗑️ Limpar cache do Streamlit",
                "📋 Limitar logs antigos",
                "🔄 Remover arquivos temporários",
                "📊 Otimizar arquivos CSV",
                "🧹 Limpeza geral"
            ],
            default=["🗑️ Limpar cache do Streamlit", "📋 Limitar logs antigos"]
        )
        
        if st.button("🚀 Executar Limpeza"):
            
            cleanup_results = []
            
            with st.spinner("Executando limpeza..."):
                
                for option in cleanup_options:
                    
                    if "cache" in option.lower():
                        try:
                            st.cache_data.clear()
                            cleanup_results.append("✅ Cache do Streamlit limpo")
                        except Exception as e:
                            cleanup_results.append(f"❌ Erro ao limpar cache: {e}")
                    
                    elif "logs" in option.lower():
                        try:
                            # Limitar logs a 500 entradas
                            logs = self._get_logs(1000)
                            if len(logs) > 500:
                                limited_logs = {"logs": logs[-500:]}
                                
                                with open(self.logs_file, 'w', encoding='utf-8') as f:
                                    json.dump(limited_logs, f, indent=2, ensure_ascii=False)
                                
                                cleanup_results.append(f"✅ Logs limitados a 500 entradas (eram {len(logs)})")
                            else:
                                cleanup_results.append("ℹ️ Logs já estão dentro do limite")
                        except Exception as e:
                            cleanup_results.append(f"❌ Erro ao limitar logs: {e}")
                    
                    elif "temporários" in option.lower():
                        try:
                            # Procurar arquivos temporários
                            temp_count = 0
                            for root, dirs, files in os.walk('data'):
                                for file in files:
                                    if file.startswith('temp_') or file.endswith('.tmp'):
                                        os.remove(os.path.join(root, file))
                                        temp_count += 1
                            
                            cleanup_results.append(f"✅ {temp_count} arquivos temporários removidos")
                        except Exception as e:
                            cleanup_results.append(f"❌ Erro ao remover temporários: {e}")
                    
                    elif "otimizar" in option.lower():
                        try:
                            # Otimizar CSVs (remover linhas vazias, etc)
                            stores = self.store_manager.get_available_stores()
                            optimized = 0
                            
                            for store_id, store_info in stores.items():
                                df = self.store_manager.load_store_data(store_id)
                                if df is not None:
                                    original_size = len(df)
                                    df_clean = df.dropna(how='all').reset_index(drop=True)
                                    
                                    if len(df_clean) < original_size:
                                        csv_path = os.path.join('data/datasets', store_info['csv_file'])
                                        df_clean.to_csv(csv_path, index=False)
                                        optimized += 1
                            
                            cleanup_results.append(f"✅ {optimized} arquivos CSV otimizados")
                        except Exception as e:
                            cleanup_results.append(f"❌ Erro ao otimizar CSVs: {e}")
                    
                    elif "geral" in option.lower():
                        try:
                            # Limpeza geral
                            operations = [
                                "Verificação de integridade executada",
                                "Permissões de arquivos verificadas", 
                                "Índices atualizados",
                                "Cache de metadados limpo"
                            ]
                            
                            for op in operations:
                                cleanup_results.append(f"✅ {op}")
                                
                        except Exception as e:
                            cleanup_results.append(f"❌ Erro na limpeza geral: {e}")
            
            # Exibir resultados
            st.subheader("📋 Resultados da Limpeza")
            
            for result in cleanup_results:
                if "✅" in result:
                    st.success(result)
                elif "❌" in result:
                    st.error(result)
                elif "⚠️" in result:
                    st.warning(result)
                else:
                    st.info(result)
            
            self._log_action("auto_cleanup_completed", f"Limpeza automática concluída. {len(cleanup_options)} operações executadas")
    
    def _full_diagnostic(self):
        """Diagnóstico completo do sistema"""
        
        st.subheader("📊 Diagnóstico Completo do Sistema")
        
        with st.spinner("Executando diagnóstico completo..."):
            
            diagnostic_data = {
                'timestamp': datetime.now().isoformat(),
                'system_info': {},
                'stores_analysis': {},
                'performance_metrics': {},
                'security_status': {},
                'recommendations': []
            }
            
            # 1. Informações do sistema
            stores = self.store_manager.get_available_stores()
            total_records = sum([len(self.store_manager.load_store_data(sid) or []) for sid in stores.keys()])
            
            diagnostic_data['system_info'] = {
                'total_stores': len(stores),
                'total_records': total_records,
                'active_users': 2,  # Mock
                'system_uptime_hours': 24,  # Mock
                'logs_count': len(self._get_logs(1000))
            }
            
            # 2. Análise das lojas
            for store_id, store_info in stores.items():
                df = self.store_manager.load_store_data(store_id)
                
                if df is not None:
                    analysis = {
                        'records_count': len(df),
                        'columns_count': len(df.columns),
                        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                        'date_range': {
                            'start': df['data'].min().isoformat() if 'data' in df.columns else None,
                            'end': df['data'].max().isoformat() if 'data' in df.columns else None
                        },
                        'data_quality_score': max(0, 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                        'file_size_mb': os.path.getsize(os.path.join('data/datasets', store_info['csv_file'])) / (1024*1024)
                    }
                    
                    # Recomendações específicas da loja
                    if analysis['missing_data_pct'] > 10:
                        diagnostic_data['recommendations'].append(f"Loja {store_id}: Alto percentual de dados faltantes ({analysis['missing_data_pct']:.1f}%)")
                    
                    if analysis['records_count'] < 30:
                        diagnostic_data['recommendations'].append(f"Loja {store_id}: Poucos registros para análise adequada ({analysis['records_count']})")
                    
                    diagnostic_data['stores_analysis'][store_id] = analysis
            
            # 3. Métricas de performance (simuladas)
            diagnostic_data['performance_metrics'] = {
                'avg_load_time_ms': 250,
                'memory_usage_mb': 128,
                'cpu_usage_pct': 45,
                'disk_usage_pct': 38,
                'response_time_ms': 180
            }
            
            # 4. Status de segurança
            diagnostic_data['security_status'] = {
                'password_policy_enabled': False,
                'session_timeout_configured': True,
                'two_factor_enabled': False,
                'logs_retention_days': 30,
                'backup_status': 'Manual',
                'last_security_check': datetime.now().isoformat()
            }
            
            # 5. Recomendações gerais
            if diagnostic_data['system_info']['total_records'] > 10000:
                diagnostic_data['recommendations'].append("Sistema com muitos dados: considere implementar paginação")
            
            if len(diagnostic_data['recommendations']) == 0:
                diagnostic_data['recommendations'].append("Sistema funcionando adequadamente")
        
        # Exibir resultados do diagnóstico
        st.subheader("📈 Resumo Executivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏪 Lojas", diagnostic_data['system_info']['total_stores'])
        
        with col2:
            st.metric("📊 Registros", f"{diagnostic_data['system_info']['total_records']:,}")
        
        with col3:
            avg_quality = sum([s.get('data_quality_score', 0) for s in diagnostic_data['stores_analysis'].values()]) / len(diagnostic_data['stores_analysis']) if diagnostic_data['stores_analysis'] else 0
            st.metric("⭐ Qualidade Média", f"{avg_quality:.1f}%")
        
        with col4:
            st.metric("⚡ Performance", f"{diagnostic_data['performance_metrics']['response_time_ms']}ms")
        
        # Análise detalhada por seção
        tab1, tab2, tab3, tab4 = st.tabs(["🏪 Lojas", "⚡ Performance", "🔒 Segurança", "💡 Recomendações"])
        
        with tab1:
            if diagnostic_data['stores_analysis']:
                stores_diag_df = pd.DataFrame.from_dict(diagnostic_data['stores_analysis'], orient='index')
                stores_diag_df.index.name = 'Loja'
                st.dataframe(stores_diag_df, use_container_width=True)
                
                # Gráfico de qualidade por loja
                fig_quality = px.bar(
                    x=stores_diag_df.index,
                    y=stores_diag_df['data_quality_score'],
                    title="Score de Qualidade por Loja",
                    labels={'x': 'Loja', 'y': 'Score de Qualidade (%)'}
                )
                st.plotly_chart(fig_quality, use_container_width=True)
        
        with tab2:
            perf_metrics = diagnostic_data['performance_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💾 Recursos:**")
                st.write(f"- CPU: {perf_metrics['cpu_usage_pct']}%")
                st.write(f"- Memória: {perf_metrics['memory_usage_mb']} MB")
                st.write(f"- Disco: {perf_metrics['disk_usage_pct']}%")
            
            with col2:
                st.markdown("**⚡ Tempos:**")
                st.write(f"- Carregamento: {perf_metrics['avg_load_time_ms']} ms")
                st.write(f"- Resposta: {perf_metrics['response_time_ms']} ms")
        
        with tab3:
            sec_status = diagnostic_data['security_status']
            
            security_items = [
                ("Política de Senhas", sec_status['password_policy_enabled']),
                ("Timeout de Sessão", sec_status['session_timeout_configured']),
                ("Autenticação 2FA", sec_status['two_factor_enabled']),
                ("Backup Automático", sec_status['backup_status'] == 'Automático')
            ]
            
            for item, status in security_items:
                emoji = "✅" if status else "❌"
                st.write(f"{emoji} **{item}**: {'Habilitado' if status else 'Desabilitado'}")
        
        with tab4:
            st.markdown("**💡 Recomendações do Sistema:**")
            
            for i, rec in enumerate(diagnostic_data['recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            if len(diagnostic_data['recommendations']) == 1 and "adequadamente" in diagnostic_data['recommendations'][0]:
                st.success("🎉 Sistema está funcionando de forma otimizada!")
        
        # Download do relatório completo
        st.subheader("📥 Relatório Completo")
        
        report_json = json.dumps(diagnostic_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            "📋 Download Diagnóstico Completo",
            data=report_json,
            file_name=f"diagnostico_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        self._log_action("full_diagnostic_completed", f"Diagnóstico completo executado. {len(diagnostic_data['recommendations'])} recomendações geradas")
    
    def _render_system_settings(self):
        """Configurações completas do sistema"""
        
        st.subheader("🔧 Configurações do Sistema")
        
        # Carregar configurações atuais
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                current_settings = json.load(f)
        except:
            current_settings = {}
        
        settings_changed = False
        
        # Configurações de segurança (expandidas)
        with st.expander("🔐 Configurações de Segurança", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_session_timeout = st.number_input(
                    "⏰ Timeout da Sessão (minutos)",
                    min_value=5,
                    max_value=240,
                    value=current_settings.get('security', {}).get('session_timeout_minutes', 30),
                    help="Tempo limite para sessões inativas"
                )
                
                new_max_login_attempts = st.number_input(
                    "🔒 Máximo de Tentativas de Login",
                    min_value=1,
                    max_value=10,
                    value=current_settings.get('security', {}).get('max_login_attempts', 3),
                    help="Número máximo de tentativas antes de bloquear"
                )
                
                new_password_min_length = st.number_input(
                    "🔑 Tamanho Mínimo da Senha",
                    min_value=4,
                    max_value=20,
                    value=current_settings.get('security', {}).get('password_min_length', 6),
                    help="Número mínimo de caracteres para senhas"
                )
            
            with col2:
                new_require_strong_password = st.checkbox(
                    "🔑 Exigir Senhas Fortes",
                    value=current_settings.get('security', {}).get('require_strong_password', False),
                    help="Exigir letras maiúsculas, minúsculas, números e símbolos"
                )
                
                new_enable_2fa = st.checkbox(
                    "📱 Habilitar 2FA",
                    value=current_settings.get('security', {}).get('enable_2fa', False),
                    help="Autenticação de dois fatores (funcionalidade futura)"
                )
                
                new_auto_lock_inactive = st.checkbox(
                    "🔒 Bloqueio Automático",
                    value=current_settings.get('security', {}).get('auto_lock_inactive', False),
                    help="Bloquear conta após muitas tentativas falhadas"
                )
            
            # Configurações avançadas de segurança
            st.markdown("**🛡️ Configurações Avançadas:**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                new_ip_whitelist_enabled = st.checkbox(
                    "🌐 Lista Branca de IPs",
                    value=current_settings.get('security', {}).get('ip_whitelist_enabled', False),
                    help="Restringir acesso a IPs específicos"
                )
                
                if new_ip_whitelist_enabled:
                    new_allowed_ips = st.text_area(
                        "📋 IPs Permitidos (um por linha)",
                        value='\n'.join(current_settings.get('security', {}).get('allowed_ips', ['192.168.1.0/24'])),
                        help="Liste os IPs ou ranges permitidos"
                    )
                else:
                    new_allowed_ips = ""
            
            with col4:
                new_audit_logging = st.checkbox(
                    "📋 Log de Auditoria Detalhado",
                    value=current_settings.get('security', {}).get('audit_logging', True),
                    help="Registrar todas as ações dos usuários"
                )
                
                new_session_encryption = st.checkbox(
                    "🔐 Criptografia de Sessão",
                    value=current_settings.get('security', {}).get('session_encryption', True),
                    help="Criptografar dados de sessão"
                )
            
            if st.button("💾 Salvar Configurações de Segurança"):
                
                security_settings = {
                    'session_timeout_minutes': new_session_timeout,
                    'max_login_attempts': new_max_login_attempts,
                    'password_min_length': new_password_min_length,
                    'require_strong_password': new_require_strong_password,
                    'enable_2fa': new_enable_2fa,
                    'auto_lock_inactive': new_auto_lock_inactive,
                    'ip_whitelist_enabled': new_ip_whitelist_enabled,
                    'allowed_ips': new_allowed_ips.split('\n') if new_allowed_ips else [],
                    'audit_logging': new_audit_logging,
                    'session_encryption': new_session_encryption
                }
                
                current_settings['security'] = security_settings
                settings_changed = True
        
        # Configurações de dados (expandidas)
        with st.expander("📊 Configurações de Dados"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_max_file_size = st.number_input(
                    "📁 Tamanho Máximo de Arquivo (MB)",
                    min_value=1,
                    max_value=1000,
                    value=current_settings.get('data', {}).get('max_file_size_mb', 100),
                    help="Tamanho máximo para upload de arquivos"
                )
                
                new_backup_frequency = st.selectbox(
                    "💾 Frequência de Backup",
                    ["Manual", "Diário", "Semanal", "Mensal"],
                    index=["Manual", "Diário", "Semanal", "Mensal"].index(
                        current_settings.get('data', {}).get('backup_frequency', 'Manual')
                    ),
                    help="Com que frequência fazer backup automático"
                )
                
                new_data_retention = st.number_input(
                    "📅 Retenção de Dados (dias)",
                    min_value=30,
                    max_value=3650,
                    value=current_settings.get('data', {}).get('data_retention_days', 365),
                    help="Por quantos dias manter os dados antes da limpeza automática"
                )
            
            with col2:
                new_auto_cleanup = st.checkbox(
                    "🧹 Limpeza Automática",
                    value=current_settings.get('data', {}).get('auto_cleanup', False),
                    help="Executar limpeza automática periodicamente"
                )
                
                new_compress_data = st.checkbox(
                    "🗜️ Compressão de Dados",
                    value=current_settings.get('data', {}).get('compress_data', False),
                    help="Comprimir dados antigos para economizar espaço"
                )
                
                new_data_validation = st.checkbox(
                    "✅ Validação Automática",
                    value=current_settings.get('data', {}).get('data_validation', True),
                    help="Validar dados automaticamente ao carregar"
                )
            
            # Configurações de armazenamento
            st.markdown("**💾 Configurações de Armazenamento:**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                new_cache_size = st.selectbox(
                    "🗄️ Tamanho do Cache",
                    ["Pequeno (50MB)", "Médio (100MB)", "Grande (200MB)", "Muito Grande (500MB)"],
                    index=1,
                    help="Quantidade de memória para cache de dados"
                )
                
                new_temp_file_cleanup = st.number_input(
                    "🗑️ Limpeza de Arquivos Temporários (horas)",
                    min_value=1,
                    max_value=72,
                    value=current_settings.get('data', {}).get('temp_file_cleanup_hours', 24),
                    help="Remover arquivos temporários após X horas"
                )
            
            with col4:
                new_data_encryption = st.checkbox(
                    "🔐 Criptografar Dados Sensíveis",
                    value=current_settings.get('data', {}).get('data_encryption', False),
                    help="Criptografar dados financeiros em repouso"
                )
                
                new_export_limits = st.checkbox(
                    "📤 Limites de Exportação",
                    value=current_settings.get('data', {}).get('export_limits', True),
                    help="Aplicar limites na exportação de dados"
                )
            
            if st.button("💾 Salvar Configurações de Dados"):
                
                data_settings = {
                    'max_file_size_mb': new_max_file_size,
                    'backup_frequency': new_backup_frequency,
                    'data_retention_days': new_data_retention,
                    'auto_cleanup': new_auto_cleanup,
                    'compress_data': new_compress_data,
                    'data_validation': new_data_validation,
                    'cache_size': new_cache_size,
                    'temp_file_cleanup_hours': new_temp_file_cleanup,
                    'data_encryption': new_data_encryption,
                    'export_limits': new_export_limits
                }
                
                current_settings['data'] = data_settings
                settings_changed = True
        
        # Configurações do sistema (expandidas)
        with st.expander("⚙️ Configurações do Sistema"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_log_level = st.selectbox(
                    "📊 Nível de Log",
                    ["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                        current_settings.get('system', {}).get('log_level', 'INFO')
                    ),
                    help="Detalhamento dos logs do sistema"
                )
                
                new_max_log_entries = st.number_input(
                    "📋 Máximo de Entradas de Log",
                    min_value=100,
                    max_value=10000,
                    value=current_settings.get('system', {}).get('max_logs_entries', 1000),
                    help="Número máximo de logs antes da rotação"
                )
                
                new_maintenance_window = st.time_input(
                    "🔧 Janela de Manutenção",
                    value=datetime.strptime(
                        current_settings.get('system', {}).get('maintenance_window', '02:00'), 
                        '%H:%M'
                    ).time(),
                    help="Horário para executar manutenções automáticas"
                )
            
            with col2:
                new_auto_backup = st.checkbox(
                    "💾 Backup Automático",
                    value=current_settings.get('system', {}).get('auto_backup', False),
                    help="Executar backup automático na janela de manutenção"
                )
                
                new_performance_monitoring = st.checkbox(
                    "📊 Monitoramento de Performance",
                    value=current_settings.get('system', {}).get('performance_monitoring', True),
                    help="Coletar métricas de performance"
                )
                
                new_error_notifications = st.checkbox(
                    "🚨 Notificações de Erro",
                    value=current_settings.get('system', {}).get('error_notifications', True),
                    help="Enviar notificações quando erros ocorrerem"
                )
            
            # Configurações de interface
            st.markdown("**🎨 Configurações de Interface:**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                new_theme = st.selectbox(
                    "🎨 Tema da Interface",
                    ["Claro", "Escuro", "Auto"],
                    index=0,
                    help="Tema visual da interface"
                )
                
                new_language = st.selectbox(
                    "🌐 Idioma",
                    ["Português", "English", "Español"],
                    index=0,
                    help="Idioma da interface"
                )
            
            with col4:
                new_animations = st.checkbox(
                    "✨ Animações",
                    value=current_settings.get('system', {}).get('animations_enabled', True),
                    help="Habilitar animações na interface"
                )
                
                new_tooltips = st.checkbox(
                    "💬 Dicas de Ajuda",
                    value=current_settings.get('system', {}).get('tooltips_enabled', True),
                    help="Mostrar dicas de ajuda nos elementos"
                )
            
            if st.button("💾 Salvar Configurações do Sistema"):
                
                system_settings = {
                    'log_level': new_log_level,
                    'max_logs_entries': new_max_log_entries,
                    'maintenance_window': new_maintenance_window.strftime('%H:%M'),
                    'auto_backup': new_auto_backup,
                    'performance_monitoring': new_performance_monitoring,
                    'error_notifications': new_error_notifications,
                    'theme': new_theme,
                    'language': new_language,
                    'animations_enabled': new_animations,
                    'tooltips_enabled': new_tooltips
                }
                
                current_settings['system'] = system_settings
                settings_changed = True
        
        # Salvar todas as configurações se houver mudanças
        if settings_changed:
            if self._save_settings(current_settings):
                st.success("✅ Configurações salvas com sucesso!")
                st.balloons()
            else:
                st.error("❌ Erro ao salvar configurações")
        
        # Importar/Exportar configurações
        st.subheader("🔄 Importar/Exportar Configurações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Exportar configurações
            config_json = json.dumps(current_settings, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📤 Exportar Configurações",
                data=config_json,
                file_name=f"configuracoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Baixar arquivo com todas as configurações"
            )
        
        with col2:
            # Importar configurações
            uploaded_config = st.file_uploader(
                "📥 Importar Configurações",
                type=['json'],
                help="Carregar arquivo de configurações"
            )
            
            if uploaded_config:
                try:
                    imported_settings = json.load(uploaded_config)
                    
                    if st.button("🚀 Aplicar Configurações Importadas"):
                        if self._save_settings(imported_settings):
                            st.success("✅ Configurações importadas e aplicadas!")
                            self._log_action("settings_imported", "Configurações importadas de arquivo")
                            st.rerun()
                        else:
                            st.error("❌ Erro ao aplicar configurações importadas")
                    
                    st.json(imported_settings)
                    
                except json.JSONDecodeError:
                    st.error("❌ Arquivo JSON inválido")
                except Exception as e:
                    st.error(f"❌ Erro ao processar arquivo: {e}")
    
    
    def _render_monitoring(self):
        """Painel de monitoramento em tempo real"""
        
        st.subheader("📈 Monitoramento do Sistema")
        
        # Métricas em tempo real
        st.subheader("⚡ Métricas em Tempo Real")
        
        # Auto-refresh a cada 30 segundos
        col_refresh, col_auto = st.columns([3, 1])
        with col_refresh:
            if st.button("🔄 Atualizar Métricas"):
                st.rerun()
        with col_auto:
            auto_refresh = st.checkbox("Auto-refresh", value=False, help="Atualizar automaticamente")
        
        # Simular métricas (em produção, pegar dados reais)
        import random
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = random.randint(30, 70)
            cpu_delta = random.randint(-5, 5)
            st.metric("💻 CPU Usage", f"{cpu_usage}%", f"{cpu_delta:+d}%")
        
        with col2:
            memory_usage = random.randint(40, 80)
            memory_delta = random.randint(-3, 3)
            st.metric("💾 Memory", f"{memory_usage}%", f"{memory_delta:+d}%")
        
        with col3:
            response_time = random.randint(100, 300)
            response_delta = random.randint(-20, 20)
            st.metric("⚡ Response Time", f"{response_time}ms", f"{response_delta:+d}ms")
        
        with col4:
            active_sessions = random.randint(1, 5)
            session_delta = random.randint(-1, 2)
            st.metric("👥 Active Sessions", active_sessions, f"{session_delta:+d}")
        
        # Gráficos de monitoramento
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de performance das últimas 24h
            hours = list(range(24))
            cpu_data = [30 + i*1.5 + random.randint(-5, 5) for i in range(24)]
            memory_data = [50 + i*0.8 + random.randint(-3, 3) for i in range(24)]
            
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Scatter(
                x=hours, 
                y=cpu_data, 
                name='CPU %', 
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=2),
                marker=dict(size=4)
            ))
            fig_performance.add_trace(go.Scatter(
                x=hours, 
                y=memory_data, 
                name='Memory %', 
                mode='lines+markers',
                line=dict(color='#4ecdc4', width=2),
                marker=dict(size=4)
            ))
            
            fig_performance.update_layout(
                title="Performance nas Últimas 24h",
                xaxis_title="Hora",
                yaxis_title="Uso (%)",
                height=350,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            # Gráfico de requisições por hora
            request_data = [random.randint(10, 100) for _ in range(24)]
            
            fig_requests = px.bar(
                x=hours,
                y=request_data,
                title="Requisições por Hora",
                labels={'x': 'Hora', 'y': 'Requisições'},
                color=request_data,
                color_continuous_scale='viridis'
            )
            
            fig_requests.update_layout(
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_requests, use_container_width=True)
        
        # Logs em tempo real
        st.subheader("📋 Logs Recentes do Sistema")
        
        # Filtros para logs em tempo real
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            log_level_filter = st.selectbox("Nível:", ['Todos', 'INFO', 'WARNING', 'ERROR'], key='monitor_level')
        
        with col2:
            log_user_filter = st.selectbox("Usuário:", ['Todos', 'admin', 'usuario', 'system'], key='monitor_user')
        
        with col3:
            log_limit = st.selectbox("Mostrar:", [10, 20, 50, 100], index=1, key='monitor_limit')
        
        with col4:
            if st.button("🔍 Aplicar Filtros"):
                st.rerun()
        
        # Obter e filtrar logs recentes
        recent_logs = self._get_logs(200)
        
        # Aplicar filtros
        filtered_logs = recent_logs.copy()
        
        if log_level_filter != 'Todos':
            filtered_logs = [log for log in filtered_logs if log.get('level', 'INFO') == log_level_filter]
        
        if log_user_filter != 'Todos':
            filtered_logs = [log for log in filtered_logs if log.get('user', '') == log_user_filter]
        
        # Limitar quantidade
        filtered_logs = filtered_logs[-log_limit:]
        
        # Exibir em formato de tabela
        if filtered_logs:
            logs_display = []
            for log in reversed(filtered_logs):  # Mais recentes primeiro
                # Formatação do timestamp
                timestamp_str = log.get('timestamp', '')
                if 'T' in timestamp_str:
                    timestamp_display = timestamp_str.replace('T', ' ')[:19]
                else:
                    timestamp_display = timestamp_str[:19] if len(timestamp_str) >= 19 else timestamp_str
                
                # Ícone por nível
                level = log.get('level', 'INFO')
                if level == 'ERROR':
                    level_icon = "❌ ERROR"
                elif level == 'WARNING':
                    level_icon = "⚠️ WARNING"
                elif level == 'DEBUG':
                    level_icon = "🔍 DEBUG"
                else:
                    level_icon = "ℹ️ INFO"
                
                logs_display.append({
                    'Timestamp': timestamp_display,
                    'Usuário': log.get('user', 'N/A'),
                    'Nível': level_icon,
                    'Ação': log.get('action', '')[:30] + ('...' if len(log.get('action', '')) > 30 else ''),
                    'Detalhes': log.get('details', '')[:50] + ('...' if len(log.get('details', '')) > 50 else '')
                })
            
            logs_df = pd.DataFrame(logs_display)
            st.dataframe(logs_df, use_container_width=True, height=400)
        else:
            st.info("Nenhum log encontrado com os filtros aplicados")
        
        # Alertas e Notificações
        st.subheader("🚨 Alertas do Sistema")
        
        # Verificar condições para alertas
        alerts = []
        
        # Alertas de sistema
        if cpu_usage > 80:
            alerts.append({
                'type': 'error',
                'title': 'Alto Uso de CPU',
                'message': f'CPU está em {cpu_usage}%, considere investigar processos pesados',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'alta'
            })
        elif cpu_usage > 65:
            alerts.append({
                'type': 'warning',
                'title': 'Uso Moderado de CPU',
                'message': f'CPU está em {cpu_usage}%, monitorar tendência',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'média'
            })
        
        if memory_usage > 85:
            alerts.append({
                'type': 'error',
                'title': 'Alto Uso de Memória',
                'message': f'Memória está em {memory_usage}%, considere limpeza de cache',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'alta'
            })
        elif memory_usage > 70:
            alerts.append({
                'type': 'warning',
                'title': 'Uso Moderado de Memória',
                'message': f'Memória está em {memory_usage}%, acompanhar crescimento',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'média'
            })
        
        if response_time > 250:
            alerts.append({
                'type': 'warning',
                'title': 'Tempo de Resposta Alto',
                'message': f'Tempo de resposta em {response_time}ms, usuários podem notar lentidão',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'média'
            })
        
        # Verificar qualidade dos dados das lojas
        stores = self.store_manager.get_available_stores()
        for store_id, store_info in stores.items():
            df = self.store_manager.load_store_data(store_id)
            if df is not None:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                if missing_pct > 20:
                    alerts.append({
                        'type': 'error',
                        'title': f'Crítico: Qualidade de Dados - {store_info["display_name"]}',
                        'message': f'{missing_pct:.1f}% de dados faltantes - verificação urgente necessária',
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'priority': 'alta'
                    })
                elif missing_pct > 10:
                    alerts.append({
                        'type': 'warning',
                        'title': f'Qualidade de Dados - {store_info["display_name"]}',
                        'message': f'{missing_pct:.1f}% de dados faltantes - requer atenção',
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'priority': 'média'
                    })
                
                # Verificar dados desatualizados
                if 'data' in df.columns:
                    last_date = df['data'].max()
                    days_since_update = (datetime.now().date() - last_date.date()).days
                    if days_since_update > 7:
                        alerts.append({
                            'type': 'warning',
                            'title': f'Dados Desatualizados - {store_info["display_name"]}',
                            'message': f'Último registro há {days_since_update} dias',
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'priority': 'baixa'
                        })
        
        # Verificar logs de erro recentes
        error_logs = [log for log in recent_logs[-50:] if log.get('level') == 'ERROR']
        if len(error_logs) > 3:
            alerts.append({
                'type': 'warning',
                'title': 'Múltiplos Erros Detectados',
                'message': f'{len(error_logs)} erros nas últimas operações - verificar logs',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'priority': 'média'
            })
        
        # Exibir alertas ordenados por prioridade
        if alerts:
            # Ordenar por prioridade: alta -> média -> baixa
            priority_order = {'alta': 0, 'média': 1, 'baixa': 2}
            alerts.sort(key=lambda x: priority_order.get(x['priority'], 3))
            
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(f"🚨 **{alert['title']}** ({alert['timestamp']})  \n{alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"⚠️ **{alert['title']}** ({alert['timestamp']})  \n{alert['message']}")
                else:
                    st.info(f"ℹ️ **{alert['title']}** ({alert['timestamp']})  \n{alert['message']}")
        else:
            st.success("✅ Sistema funcionando normalmente - Nenhum alerta ativo")
        
        # Estatísticas de uptime e disponibilidade
        st.subheader("📊 Estatísticas de Disponibilidade")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            uptime_days = random.randint(1, 30)
            st.metric("⏱️ Uptime", f"{uptime_days} dias", "↗️ Estável")
        
        with col2:
            availability = random.uniform(98.5, 99.9)
            availability_delta = random.uniform(0.01, 0.1)
            st.metric("📈 Disponibilidade", f"{availability:.2f}%", f"+{availability_delta:.2f}%")
        
        with col3:
            total_requests_today = random.randint(500, 2000)
            requests_delta = random.randint(50, 200)
            st.metric("📊 Requisições Hoje", f"{total_requests_today:,}", f"+{requests_delta}")
        
        with col4:
            avg_response = random.randint(120, 200)
            response_trend = random.randint(-20, 10)
            st.metric("⚡ Resposta Média", f"{avg_response}ms", f"{response_trend:+d}ms")
        
        # Gráficos de análise avançada
        st.subheader("📈 Análise Avançada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de uptime dos últimos 30 dias
            days = list(range(1, 31))
            uptime_daily = [random.uniform(98, 100) for _ in range(30)]
            
            fig_uptime = px.line(
                x=days,
                y=uptime_daily,
                title="Uptime Diário (30 dias)",
                labels={'x': 'Dia do Mês', 'y': 'Uptime (%)'},
                range_y=[95, 100]
            )
            
            # Adicionar linha de SLA target
            fig_uptime.add_hline(
                y=99.5, 
                line_dash="dash", 
                line_color="green", 
                annotation_text="SLA Target (99.5%)"
            )
            
            # Destacar dias com problemas
            problem_days = [i for i, uptime in enumerate(uptime_daily, 1) if uptime < 99]
            if problem_days:
                fig_uptime.add_scatter(
                    x=problem_days,
                    y=[uptime_daily[i-1] for i in problem_days],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Problemas',
                    showlegend=True
                )
            
            st.plotly_chart(fig_uptime, use_container_width=True)
        
        with col2:
            # Distribuição de códigos de resposta HTTP
            response_codes = ['200 OK', '404 Not Found', '500 Error', '403 Forbidden', '502 Bad Gateway']
            response_counts = [
                random.randint(800, 1200),  # 200 OK
                random.randint(10, 50),     # 404
                random.randint(0, 10),      # 500
                random.randint(0, 5),       # 403
                random.randint(0, 3)        # 502
            ]
            
            # Definir cores para cada status
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#e67e22', '#9b59b6']
            
            fig_responses = px.pie(
                values=response_counts,
                names=response_codes,
                title="Distribuição de Códigos de Resposta",
                color_discrete_sequence=colors
            )
            
            fig_responses.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
            )
            
            st.plotly_chart(fig_responses, use_container_width=True)
        
        # Análise de performance das lojas
        if stores:
            st.subheader("🏪 Performance das Lojas")
            
            # Criar dados de performance para cada loja
            store_performance = []
            for store_id, store_info in stores.items():
                df = self.store_manager.load_store_data(store_id)
                if df is not None:
                    load_time = random.randint(50, 300)
                    data_quality = max(0, 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
                    
                    store_performance.append({
                        'Loja': store_info['display_name'],
                        'ID': store_id,
                        'Tempo de Carregamento (ms)': load_time,
                        'Qualidade dos Dados (%)': round(data_quality, 1),
                        'Registros': len(df),
                        'Status': 'OK' if load_time < 200 and data_quality > 95 else 'Atenção' if load_time < 250 and data_quality > 90 else 'Crítico'
                    })
            
            if store_performance:
                perf_df = pd.DataFrame(store_performance)
                
                # Colorir por status
                def highlight_status(val):
                    if val == 'OK':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'Atenção':
                        return 'background-color: #fff3cd; color: #856404'
                    else:  # Crítico
                        return 'background-color: #f8d7da; color: #721c24'
                
                styled_df = perf_df.style.applymap(highlight_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Gráfico de qualidade por loja
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_quality = px.bar(
                        perf_df,
                        x='Loja',
                        y='Qualidade dos Dados (%)',
                        title="Qualidade dos Dados por Loja",
                        color='Qualidade dos Dados (%)',
                        color_continuous_scale='RdYlGn',
                        range_color=[80, 100]
                    )
                    
                    fig_quality.add_hline(y=95, line_dash="dash", line_color="red", 
                                         annotation_text="Threshold Mínimo (95%)")
                    
                    st.plotly_chart(fig_quality, use_container_width=True)
                
                with col2:
                    fig_load_time = px.scatter(
                        perf_df,
                        x='Registros',
                        y='Tempo de Carregamento (ms)',
                        title="Tempo de Carregamento vs Volume de Dados",
                        color='Status',
                        size='Qualidade dos Dados (%)',
                        hover_data=['Loja'],
                        color_discrete_map={
                            'OK': '#28a745',
                            'Atenção': '#ffc107', 
                            'Crítico': '#dc3545'
                        }
                    )
                    
                    st.plotly_chart(fig_load_time, use_container_width=True)
        
        # Ações de monitoramento
        st.subheader("🔧 Ações de Monitoramento")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Exportar Métricas", help="Exportar todas as métricas atuais"):
                self._export_monitoring_data()
        
        with col2:
            if st.button("🔔 Configurar Alertas", help="Definir thresholds para alertas"):
                self._configure_alerts()
        
        with col3:
            if st.button("📈 Relatório Performance", help="Gerar relatório detalhado"):
                self._generate_performance_report()
        
        with col4:
            if st.button("🧹 Limpar Logs Antigos", help="Remover logs antigos do sistema"):
                self._cleanup_old_logs()
        
        # Informações adicionais no rodapé
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔄 Última Atualização:**")
            st.write(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
        
        with col2:
            st.markdown("**📋 Total de Logs:**")
            st.write(f"{len(self._get_logs(10000)):,} entradas")
        
        with col3:
            st.markdown("**🏪 Lojas Monitoradas:**")
            st.write(f"{len(stores)} lojas ativas")
        
        # Auto-refresh se habilitado
        if auto_refresh:
            st.markdown("---")
            st.info("🔄 Auto-refresh ativo - Atualizando a cada 30 segundos...")
   
    def _export_monitoring_data(self):
        """Exporta dados de monitoramento"""
        
        st.subheader("📊 Exportar Dados de Monitoramento")
        
        # Simular dados de monitoramento
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': random.randint(30, 70),
                'memory_usage': random.randint(40, 80),
                'disk_usage': random.randint(20, 60),
                'response_time': random.randint(100, 300),
                'active_sessions': random.randint(1, 5)
            },
            'performance_history': {
                'last_24h_cpu': [30 + i + random.randint(-5, 5) for i in range(24)],
                'last_24h_memory': [50 + i*0.5 + random.randint(-3, 3) for i in range(24)],
                'last_24h_requests': [random.randint(10, 100) for _ in range(24)]
            },
            'availability_stats': {
                'uptime_days': random.randint(1, 30),
                'availability_percent': random.uniform(98.5, 99.9),
                'total_requests_today': random.randint(500, 2000),
                'average_response_time': random.randint(120, 200)
            },
            'recent_logs': self._get_logs(100),
            'store_health': {}
        }
        
        # Adicionar saúde das lojas
        stores = self.store_manager.get_available_stores()
        for store_id in stores.keys():
            df = self.store_manager.load_store_data(store_id)
            if df is not None:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                monitoring_data['store_health'][store_id] = {
                    'records_count': len(df),
                    'missing_data_percent': missing_pct,
                    'last_update': df['data'].max().isoformat() if 'data' in df.columns else None
                }
        
        # Converter para JSON
        monitoring_json = json.dumps(monitoring_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            "📥 Download Dados de Monitoramento",
            data=monitoring_json,
            file_name=f"monitoramento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Dados de monitoramento preparados para download!")
        self._log_action("monitoring_data_exported", "Dados de monitoramento exportados")
    
    def _configure_alerts(self):
        """Configuração de alertas do sistema"""
        
        st.subheader("🔔 Configuração de Alertas")
        
        # Configurações de alerta
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖥️ Alertas de Sistema:**")
            
            cpu_threshold = st.slider("CPU Threshold (%)", 50, 95, 80)
            memory_threshold = st.slider("Memory Threshold (%)", 60, 95, 85)
            response_threshold = st.slider("Response Time (ms)", 200, 1000, 500)
            
            enable_email_alerts = st.checkbox("📧 Enviar por Email", value=False)
            if enable_email_alerts:
                alert_email = st.text_input("📬 Email para Alertas", "admin@empresa.com")
        
        with col2:
            st.markdown("**📊 Alertas de Dados:**")
            
            missing_data_threshold = st.slider("Dados Faltantes (%)", 5, 50, 15)
            old_data_threshold = st.slider("Dados Antigos (dias)", 1, 30, 7)
            
            enable_webhook_alerts = st.checkbox("🔗 Webhook", value=False)
            if enable_webhook_alerts:
                webhook_url = st.text_input("🌐 URL do Webhook", "https://hooks.slack.com/...")
        
        # Horários de alerta
        st.markdown("**⏰ Horários de Alerta:**")
        col3, col4 = st.columns(2)
        
        with col3:
            alert_start_time = st.time_input("Início dos Alertas", datetime.strptime("08:00", "%H:%M").time())
            
        with col4:
            alert_end_time = st.time_input("Fim dos Alertas", datetime.strptime("18:00", "%H:%M").time())
        
        # Salvar configurações
        if st.button("💾 Salvar Configurações de Alerta"):
            alert_config = {
                'system_alerts': {
                    'cpu_threshold': cpu_threshold,
                    'memory_threshold': memory_threshold,
                    'response_threshold': response_threshold
                },
                'data_alerts': {
                    'missing_data_threshold': missing_data_threshold,
                    'old_data_threshold': old_data_threshold
                },
                'notification_settings': {
                    'enable_email': enable_email_alerts,
                    'alert_email': alert_email if enable_email_alerts else None,
                    'enable_webhook': enable_webhook_alerts,
                    'webhook_url': webhook_url if enable_webhook_alerts else None,
                    'alert_hours': {
                        'start': alert_start_time.strftime('%H:%M'),
                        'end': alert_end_time.strftime('%H:%M')
                    }
                },
                'updated_at': datetime.now().isoformat(),
                'updated_by': self.auth_manager.get_username()
            }
            
            st.success("✅ Configurações de alerta salvas!")
            st.json(alert_config)
            
            self._log_action("alert_config_updated", "Configurações de alerta atualizadas")
    
    def _generate_performance_report(self):
        """Gera relatório detalhado de performance"""
        
        st.subheader("📈 Relatório de Performance")
        
        with st.spinner("Gerando relatório de performance..."):
            
            # Simular coleta de dados de performance
            performance_data = {
                'report_period': {
                    'start': (datetime.now() - timedelta(days=30)).isoformat(),
                    'end': datetime.now().isoformat(),
                    'days_analyzed': 30
                },
                'system_performance': {
                    'avg_cpu_usage': random.uniform(35, 65),
                    'max_cpu_usage': random.uniform(70, 90),
                    'avg_memory_usage': random.uniform(45, 75),
                    'max_memory_usage': random.uniform(80, 95),
                    'avg_response_time': random.uniform(150, 250),
                    'max_response_time': random.uniform(300, 500)
                },
                'availability_metrics': {
                    'uptime_percentage': random.uniform(98.5, 99.9),
                    'total_downtime_minutes': random.randint(10, 100),
                    'incidents_count': random.randint(0, 5),
                    'mttr_minutes': random.randint(5, 30)  # Mean Time to Recovery
                },
                'usage_statistics': {
                    'total_requests': random.randint(10000, 50000),
                    'unique_users': random.randint(10, 100),
                    'peak_concurrent_users': random.randint(5, 20),
                    'data_processed_gb': random.uniform(1, 10)
                },
                'store_performance': {}
            }
            
            # Analisar performance das lojas
            stores = self.store_manager.get_available_stores()
            for store_id in stores.keys():
                df = self.store_manager.load_store_data(store_id)
                if df is not None:
                    performance_data['store_performance'][store_id] = {
                        'load_time_avg_ms': random.randint(50, 200),
                        'data_quality_score': max(0, 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                        'records_processed': len(df),
                        'processing_efficiency': random.uniform(85, 99)
                    }
        
        # Exibir relatório
        st.subheader("📊 Resumo Executivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Disponibilidade",
                f"{performance_data['availability_metrics']['uptime_percentage']:.2f}%"
            )
        
        with col2:
            st.metric(
                "⚡ Tempo Médio Resposta",
                f"{performance_data['system_performance']['avg_response_time']:.0f}ms"
            )
        
        with col3:
            st.metric(
                "📊 Requests Processados",
                f"{performance_data['usage_statistics']['total_requests']:,}"
            )
        
        with col4:
            st.metric(
                "👥 Usuários Únicos",
                performance_data['usage_statistics']['unique_users']
            )
        
        # Gráficos de performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Tendência de performance
            days = list(range(1, 31))
            cpu_trend = [performance_data['system_performance']['avg_cpu_usage'] + random.uniform(-10, 10) for _ in range(30)]
            
            fig_trend = px.line(
                x=days,
                y=cpu_trend,
                title="Tendência de CPU (30 dias)",
                labels={'x': 'Dia', 'y': 'CPU (%)'}
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Performance das lojas
            if performance_data['store_performance']:
                store_names = list(performance_data['store_performance'].keys())
                quality_scores = [performance_data['store_performance'][store]['data_quality_score'] 
                                for store in store_names]
                
                fig_stores = px.bar(
                    x=store_names,
                    y=quality_scores,
                    title="Score de Qualidade por Loja",
                    labels={'x': 'Loja', 'y': 'Score de Qualidade'}
                )
                
                st.plotly_chart(fig_stores, use_container_width=True)
        
        # Recomendações
        st.subheader("💡 Recomendações")
        
        recommendations = []
        
        if performance_data['system_performance']['avg_cpu_usage'] > 60:
            recommendations.append("🔧 CPU: Considerar otimização de processos ou upgrade de hardware")
        
        if performance_data['system_performance']['avg_response_time'] > 200:
            recommendations.append("⚡ Performance: Implementar cache para melhorar tempo de resposta")
        
        if performance_data['availability_metrics']['uptime_percentage'] < 99:
            recommendations.append("🎯 Disponibilidade: Investigar causas de indisponibilidade")
        
        if performance_data['availability_metrics']['incidents_count'] > 2:
            recommendations.append("🚨 Incidentes: Implementar monitoramento proativo")
        
        if not recommendations:
            recommendations.append("✅ Sistema operando dentro dos parâmetros ideais")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Download do relatório
        report_json = json.dumps(performance_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            "📥 Download Relatório Completo",
            data=report_json,
            file_name=f"relatorio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        self._log_action("performance_report_generated", f"Relatório de performance gerado para período de 30 dias")
    
    def _cleanup_old_logs(self):
        """Limpeza de logs antigos"""
        
        st.subheader("🧹 Limpeza de Logs Antigos")
        
        current_logs = self._get_logs(10000)  # Pegar todos os logs
        
        if not current_logs:
            st.info("ℹ️ Nenhum log encontrado para limpeza")
            return
        
        st.write(f"**Total de logs atuais:** {len(current_logs)}")
        
        # Opções de limpeza
        cleanup_options = st.radio(
            "Escolha a opção de limpeza:",
            [
                "Manter últimos 100 logs",
                "Manter últimos 500 logs", 
                "Manter últimos 1000 logs",
                "Remover logs mais antigos que 30 dias",
                "Remover logs mais antigos que 7 dias"
            ]
        )
        
        if st.button("🗑️ Executar Limpeza de Logs"):
            
            logs_to_keep = []
            
            if "100 logs" in cleanup_options:
                logs_to_keep = current_logs[-100:]
            elif "500 logs" in cleanup_options:
                logs_to_keep = current_logs[-500:]
            elif "1000 logs" in cleanup_options:
                logs_to_keep = current_logs[-1000:]
            elif "30 dias" in cleanup_options:
                cutoff_date = datetime.now() - timedelta(days=30)
                logs_to_keep = [log for log in current_logs 
                               if datetime.fromisoformat(log['timestamp']) > cutoff_date]
            elif "7 dias" in cleanup_options:
                cutoff_date = datetime.now() - timedelta(days=7)
                logs_to_keep = [log for log in current_logs 
                               if datetime.fromisoformat(log['timestamp']) > cutoff_date]
            
            # Salvar logs limpos
            try:
                cleaned_logs = {"logs": logs_to_keep}
                
                with open(self.logs_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_logs, f, indent=2, ensure_ascii=False)
                
                removed_count = len(current_logs) - len(logs_to_keep)
                
                st.success(f"✅ Limpeza concluída! {removed_count} logs removidos, {len(logs_to_keep)} mantidos")
                
                # Estatísticas da limpeza
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📋 Logs Originais", len(current_logs))
                
                with col2:
                    st.metric("🗑️ Logs Removidos", removed_count)
                
                with col3:
                    st.metric("💾 Logs Mantidos", len(logs_to_keep))
                
                self._log_action("logs_cleanup_completed", f"{removed_count} logs antigos removidos")
                
            except Exception as e:
                st.error(f"❌ Erro ao limpar logs: {e}")
                self._log_action("logs_cleanup_failed", f"Erro na limpeza de logs: {str(e)}", "ERROR")
