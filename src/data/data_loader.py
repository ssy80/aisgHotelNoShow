import pandas as pd
from data.db import get_db_conn, load_df_sql
from utils.helper import setup_logging, safe_get
import logging


class DataLoader:

    def __init__(self, config: dict):

        if config is None:
            raise ValueError(f"DataLoader __init__: config cannot be None")

        self.config = config
        
        # Init logging
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Fetch data from SQLite database"""

        db_path = safe_get(self.config, 'data', 'db_path', required=True)
        table_name = safe_get(self.config, 'data', 'table_name', required=True)
        load_sql = f"SELECT * FROM {table_name};"
        
        self.logger.info(f"Loading data from {db_path}")
        
        try:
            conn = get_db_conn(db_path)
            data_df = load_df_sql(conn, load_sql)
            
            self.logger.info(f"Successfully loaded {len(data_df)} records")

            if not self.validate_data(data_df):
                raise ValueError("Data validation failed")

            return data_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the fetched data"""

        required_columns = safe_get(self.config, 'data', 'required_columns', required=True)

        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            return False
            
        if data.empty:
            self.logger.error("No data loaded from database")
            return False
            
        self.logger.info("Data validation successful")
        return True
