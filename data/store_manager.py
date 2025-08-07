import pandas as pd
import os
import json

class StoreDataManager:
    """Gerencia dados das lojas e carregamento de datasets."""
    def __init__(self):
        self.stores_config_file = 'data/stores_config.json'
        self._stores = self._load_stores_config()

    def _load_stores_config(self):
        if os.path.exists(self.stores_config_file):
            with open(self.stores_config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_available_stores(self):
        return self._stores

    def load_store_data(self, store_id):
        store_info = self._stores.get(store_id)
        if not store_info:
            return None
        csv_path = os.path.join('data/datasets', store_info['csv_file'])
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
            return df
        except Exception:
            return None
