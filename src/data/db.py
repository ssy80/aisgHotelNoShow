from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd

def get_db_conn(db)-> Engine:
    """Establishes and returns a connection to the SQLite database.
    
    Args:
        db: Path to the SQLite database file
        
    Returns:
        SQLAlchemy engine object
    """
    engine = create_engine(f"sqlite:///{db}")
    return engine


def load_df_sql(engine: Engine, load_sql: str)-> pd.DataFrame:
    """Load data to a dataframe from database by executing the sql input.
    
    Args:
        engine: SQLAlchemy engine object
        load_sql: SQL query string to execute
        
    Returns:
        Pandas DataFrame with query results
    """
    df = pd.read_sql(load_sql, engine)
    return df
