import pandas as pd
from data.db import get_db_conn, load_df_sql
from utils.helpers import setup_logging
import logging


class DataLoader:

    def __init__(self, config: dict):
        self.config = config
        
        # Init logging
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Fetch data from SQLite database"""
        db_path = self.config['data']['db_path']
        table_name = self.config['data']['table_name']
        load_sql = f"SELECT * FROM {table_name};"
        
        self.logger.info(f"Loading data from {db_path}")
        
        try:
            conn = get_db_conn(db_path)
            data_df = load_df_sql(conn, load_sql)
            
            self.logger.info(f"Successfully loaded {len(data_df)} records")
            return data_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the fetched data"""
        required_columns = self.config['data']['required_columns']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            return False
            
        if data.empty:
            self.logger.error("No data loaded from database")
            return False
            
        self.logger.info("Data validation successful")
        return True
