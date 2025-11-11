from mlpipeline import MlPipeline


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
        pipeline.run()

        #results = pipeline.run()
        #print(f"\nPipeline results: {results}")
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main();
