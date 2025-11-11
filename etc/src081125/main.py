#from db import get_db_conn
#import pandas as pd
#import seaborn as sns
#import numpy as np
from pipeline import MlPipeline


'''
def main():
    """entry point"""
    try:
        db = "data/noshow.db"
        load_sql = f"SELECT * FROM noshow ;"
        engine = get_db_conn(db)
        df = load_df_sql(engine, load_sql)

        print(df)
        # preprocessing the data
        # feature engineering
        # feature selection
        # split into training, test set
        # train model
        # predict / cross valuation
        # evaluation

    except Exception as e:
        print(f"Error: {str(e)}")
'''

def main():
    try:
        # Initialize and run pipeline
        config_file = "src/config.yaml"
        pipeline = MlPipeline(config_file)
        
        # Override config if command line arguments provided
        '''if args.algorithm:
            pipeline.config['model']['algorithm'] = args.algorithm
        if args.test_size:
            pipeline.config['data']['test_size'] = args.test_size
        '''
        pipeline.run()
        #results = pipeline.run()
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        #sys.exit(1)


if __name__ == "__main__":
    main();