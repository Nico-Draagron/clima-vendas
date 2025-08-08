# ============================================================================
# ➕ pages/register_store.py - REGISTRO DE NOVA LOJA
# ============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from data.store_manager import StoreDataManager

class RegisterStorePage:
    """Página para registro de novas lojas (Admin only)"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.store_manager = StoreDataManager()
    
    def render(self):
        """Renderiza página de registro de loja"""
        
        st.markdown("# ➕ Registrar Nova Loja")
        st.markdown("**Adicione uma nova loja ao sistema com dados de vendas**")
        
        # Verificar permissões
        if not self.auth_manager.has_permission('manage_stores'):
            st.error("❌ Acesso negado. Apenas administradores podem registrar novas lojas.")
            st.info("💡 Entre em contato com o administrador do sistema para obter acesso.")
            return
        
        # Informações sobre o processo
        with st.expander("ℹ️ Como funciona o registro de lojas", expanded=True):
            st.markdown("""
            **📋 Processo de Registro:**
            
            1. **📁 Upload do Arquivo**: Faça upload do arquivo CSV/Excel com dados de vendas
            2. **🔍 Detecção Automática**: Sistema identifica colunas de data e valor automaticamente
            3. **🌤️ Junção com Clima**: Dados são unidos com informações climáticas do período
            4. **💾 Salvamento**: Arquivo unificado é gerado e loja é registrada no sistema
            
            **📊 Requisitos do Arquivo:**
            - Deve conter uma coluna de **data** (formato: YYYY-MM-DD ou similar)
            - Deve conter uma coluna de **valor/vendas** (números)
            - Mínimo de 30 registros
            - Máximo de 10% de dados inválidos
            """)
        
        # Formulário de registro
        with st.form("register_store_form", clear_on_submit=False):
            
            st.subheader("🏪 Informações da Loja")
            
            col1, col2 = st.columns(2)
            
            with col1:
                store_name = st.text_input(
                    "📝 Nome da Loja",
                    placeholder="Ex: Loja 3",
                    help="Nome interno da loja (será usado nos arquivos)"
                )
                
                display_name = st.text_input(
                    "🏷️ Nome de Exibição",
                    placeholder="Ex: Loja Shopping Center",
                    help="Nome que será exibido na interface"
                )
            
            with col2:
                location = st.text_input(
                    "📍 Localização",
                    placeholder="Ex: Shopping Center",
                    help="Localização da loja (opcional)"
                )
                
                description = st.text_area(
                    "📄 Descrição",
                    placeholder="Ex: Loja localizada no shopping...",
                    help="Descrição adicional da loja (opcional)",
                    max_chars=200
                )
            
            st.subheader("📁 Dados de Vendas")
            
            uploaded_file = st.file_uploader(
                "📤 Escolha o arquivo com dados de vendas",
                type=['csv', 'xlsx', 'xls'],
                help="Arquivo deve conter colunas de data e valor de vendas"
            )
            
            # Opções avançadas
            with st.expander("⚙️ Opções Avançadas"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    encoding = st.selectbox(
                        "🔤 Codificação do Arquivo",
                        ['utf-8', 'latin-1', 'iso-8859-1'],
                        help="Codificação do arquivo CSV (se necessário)"
                    )
                
                with col2:
                    separator = st.selectbox(
                        "🔗 Separador CSV",
                        [',', ';', '\t'],
                        format_func=lambda x: {',' : 'Vírgula (,)', ';': 'Ponto e vírgula (;)', '\t': 'Tab'}.get(x, x),
                        help="Separador usado no arquivo CSV"
                    )
            
            # Botão de submit
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "🚀 Processar e Registrar Loja",
                    type="primary",
                    use_container_width=True
                )
        
        # Processar formulário
        if submit_button:
            if not all([store_name, display_name, uploaded_file]):
                st.error("⚠️ Preencha pelo menos os campos obrigatórios: Nome da Loja, Nome de Exibição e arquivo de dados")
                return
            
            self._process_store_registration(
                store_name=store_name,
                display_name=display_name,
                location=location,
                description=description,
                uploaded_file=uploaded_file,
                encoding=encoding,
                separator=separator
            )
    
    def _process_store_registration(self, store_name, display_name, location, description, 
                                  uploaded_file, encoding, separator):
        """Processa registro da nova loja"""
        
        st.subheader("⚙️ Processando Registro da Loja")
        
        # Validar arquivo
        with st.spinner("🔍 Validando arquivo..."):
            is_valid, message, df_sales = self.store_manager.validate_sales_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ Erro na validação: {message}")
            return
        
        st.success(f"✅ {message}")
        
        # Detectar colunas
        date_col, value_col = self.store_manager.detect_sales_columns(df_sales)
        
        st.subheader("🔍 Colunas Detectadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📅 **Coluna de Data:** {date_col}")
        
        with col2:
            st.info(f"💰 **Coluna de Valor:** {value_col}")
        
        with col3:
            st.info(f"📊 **Total de Registros:** {len(df_sales)}")
        
        # Preview dos dados
        st.subheader("👀 Preview dos Dados")
        
        preview_cols = [date_col, value_col] + [col for col in df_sales.columns if col not in [date_col, value_col]][:3]
        st.dataframe(df_sales[preview_cols].head(10), use_container_width=True)
        
        # Confirmar processamento
        if st.button("✅ Confirmar e Processar", type="primary"):
            
            # Fazer merge com dados climáticos
            st.subheader("🌤️ Unindo com Dados Climáticos")
            
            with st.spinner("🔄 Processando união dos dados..."):
                df_merged = self.store_manager.merge_sales_with_climate(
                    df_sales, date_col, value_col, store_name
                )
            
            if df_merged is None:
                st.error("❌ Erro na união dos dados. Processo cancelado.")
                return
            
            # Gerar ID da nova loja
            stores = self.store_manager.get_available_stores()
            existing_ids = [int(k.split('_')[1]) for k in stores.keys() if k.startswith('loja_')]
            new_store_id = f"loja_{max(existing_ids) + 1:03d}" if existing_ids else "loja_001"
            
            # Salvar dados unidos
            st.subheader("💾 Salvando Dados")
            
            with st.spinner("💾 Salvando arquivo unificado..."):
                if self.store_manager.save_merged_data(df_merged, new_store_id, store_name):
                    
                    # Registrar loja no sistema
                    store_data = {
                        'name': store_name,
                        'display_name': display_name,
                        'csv_file': f"{new_store_id}_dados_unificados.csv",
                        'value_column': f'valor_{store_name.lower().replace(" ", "_")}',
                        'created_date': datetime.now().strftime('%Y-%m-%d'),
                        'status': 'active',
                        'has_climate_data': True,
                        'location': location or 'Não informado',
                        'description': description or 'Sem descrição'
                    }
                    
                    if self.auth_manager.add_new_store(store_data):
                        
                        # Sucesso!
                        st.success("🎉 **Loja registrada com sucesso!**")
                        
                        # Informações finais
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"""
                            **📋 Resumo da Loja:**
                            - **ID:** {new_store_id}
                            - **Nome:** {display_name}
                            - **Arquivo:** {new_store_id}_dados_unificados.csv
                            - **Registros:** {len(df_merged)}
                            """)
                        
                        with col2:
                            st.info(f"""
                            **📊 Dados Processados:**
                            - **Período:** {df_merged['data'].min().strftime('%d/%m/%Y')} até {df_merged['data'].max().strftime('%d/%m/%Y')}
                            - **Dados Climáticos:** ✅ Incluídos
                            - **Status:** ✅ Ativo
                            """)
                        
                        # Preview dos dados finais
                        st.subheader("📊 Preview dos Dados Finais")
                        st.dataframe(df_merged.head(10), use_container_width=True)
                        
                        # Download opcional
                        csv_data = df_merged.to_csv(index=False)
                        st.download_button(
                            "📥 Download dos Dados Unificados",
                            data=csv_data,
                            file_name=f"{new_store_id}_dados_unificados.csv",
                            mime="text/csv"
                        )
                        
                        st.balloons()  # Celebração!
                        
                    else:
                        st.error("❌ Erro ao registrar loja no sistema")
                else:
                    st.error("❌ Erro ao salvar dados unificados")
