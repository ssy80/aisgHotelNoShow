import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import setup_logging
from sklearn.preprocessing import FunctionTransformer
from transformers.lowerCaseTransformer import LowercaseTransformer
from transformers.intTransformer import IntTransformer
from transformers.cleanCheckoutDayTransformer import CheckoutDayTransformer
from transformers.cleanNumAdultsTransformer import NumAdultsTransformer
from transformers.cleanFirstTimeTransformer import FirstTimeTransformer
from transformers.cleanMonthsTransformer import MonthsTransformer
from transformers.cleanPriceTransformer import PriceTransformer
from transformers.zeroToNaNTransformer import ZeroToNaNTransformer
from transformers.modePriceImputer import ModePriceImputer
from transformers.modeRoomImputer import ModeRoomImputer
from sklearn.impute import KNNImputer
from transformers.nearestMeanRoomImputer import NearestMeanRoomImputerBranch
from transformers.hasChildrenTransformer import HasChildrenTransformer
from transformers.stayedDaysTransformer import StayedDaysTransformer
from transformers.logPriceTransformer import LogPriceTransformer
from transformers.cyclicalEncoder import CyclicalEncoder
from transformers.dropColumns import DropColumns
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.preprocessor = None
    
    def remove_rows_with_n_nulls(self, X, n=3):
            """Remove rows with n or more null values"""
            return X.dropna(thresh=len(X.columns) - n + 1)
    
    def create_preprocessing_pipeline(self) -> Pipeline: #ColumnTransformer:
        """Create preprocessing pipeline based on configuration"""
        """numeric_features = self.config['preprocessing']['numeric_features']
        categorical_features = self.config['preprocessing']['categorical_features']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing']['handle_missing'])),
            ('scaler', StandardScaler() if self.config['preprocessing']['scale_numeric'] else 'passthrough')
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore') 
             if self.config['preprocessing']['encoding_strategy'] == 'onehot' 
             else LabelEncoder())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )"""
        

        # Create transformer
        drop_na_transformer = FunctionTransformer(
            self.remove_rows_with_n_nulls, 
            kw_args={'n': 3}
        )
        
        #('drop_na', drop_na_transformer),
        '''preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )'''
        #df["checkout_day"] = df["checkout_day"].apply(lambda x: (x * -1) if x < 0 else x)
        #df["num_adults"] = df["num_adults"].apply(lambda x: 1 if x == "one" else (2 if x == "two" else x))
        #df["first_time"] = df["first_time"].apply(lambda x: 1 if x == "yes" else 0)
        # custom convert months to int
        # then convert int months to cyclinical encoding.
        # ColumnTransformer for price imputation
        '''price_imputation_ct = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore'), ['room', 'branch']),
            #('numerical', 'passthrough', ['arrival_month', 'checkout_month', 'num_adults', 'num_children'])
        ])
        price_imputation_pipeline = Pipeline([
            ('select_features', price_imputation_ct),
            ('knn_imputer', KNNImputer(n_neighbors=5))
        ])'''

        categorical_cols = ["branch", "country", "room", "platform"]
        """feature_transformer = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            #('num', 'passthrough', numeric_features)
        ])
        scaler_ct = ColumnTransformer([
            ('robust', RobustScaler(), robust_features),
            ('minmax', MinMaxScaler(), minmax_features),
        ], remainder='passthrough')  # Keep all other columns untouched
        """

        lowercase_cols = ["branch", "booking_month", "arrival_month", "checkout_month", "country", "first_time", "num_adults", "platform", "room", "price"]
        int_convert_cols = ["no_show", "arrival_day", "num_children", "checkout_day", "num_adults", "first_time", "booking_month", "arrival_month", "checkout_month"]
        #cyclical_features = ["booking_month", "arrival_month", "checkout_month", "arrival_day", "checkout_day"]
        cyclical_features = {
            'arrival_month': 12,
            'checkout_month': 12,
            'booking_month': 12,
            'arrival_day': 31,
            'checkout_day': 31
        }
        robust_features = ["price"]
        minmax_features = ['num_adults', 'num_children']
        feature_transformer = ColumnTransformer([ 
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('robust', RobustScaler(), robust_features),
            ('minmax', MinMaxScaler(), minmax_features), 
            ], remainder='passthrough')

        # Create the full pipeline with both steps
        preprocessor = Pipeline([
            #('drop_na_rows', drop_na_transformer),      # Step 1: Drop rows with too many NAs
            ('drop_cols', DropColumns("booking_id")),      # Step 1: Drop rows with too many NAs
            ('lowercase', LowercaseTransformer(columns=lowercase_cols)),
            ('clean_num_adults', NumAdultsTransformer(column_name="num_adults")),
            ('clean_first_time', FirstTimeTransformer(column_name="first_time")),
            ('clean_booking_month', MonthsTransformer(column_name="booking_month")),
            ('clean_arrival_month', MonthsTransformer(column_name="arrival_month")),
            ('clean_checkout_month', MonthsTransformer(column_name="checkout_month")),
            ('clean_price', PriceTransformer(column_name="price")),

            ('impute_price', ModePriceImputer(column_name="price", cat_features=["room", "branch"])),
            ('impute_room', NearestMeanRoomImputerBranch(column_name="room", price_column="price", branch_column="branch")),
            
            ('log_price', LogPriceTransformer(column_name="price")),

            ('int_convert', IntTransformer(columns=int_convert_cols)),
            ('clean_checkout_day', CheckoutDayTransformer(column_name="checkout_day")),
            
                        #add new feature 
            ('add_has_children', HasChildrenTransformer(column_name="num_children")),
            #add stayed_days
            ('stayed_days', StayedDaysTransformer(
                column_arrival_month="arrival_month",
                column_arrival_day="arrival_day",
                column_checkout_month="checkout_month",
                column_checkout_day="checkout_day")
                ),

            ('cyclical_encode', CyclicalEncoder(columns_period_map=cyclical_features)),
            
            ('encode_features', feature_transformer)

        ])

        self.preprocessor = preprocessor
        return preprocessor
    

    def preprocess_target(self, y: pd.Series) -> pd.Series:
        """Preprocess the target variable"""
        # Example target preprocessing:
        
        if y.dtype == "float64":
            y = y.astype(int)
        
        # If target needs encoding (categorical)
        #if y.dtype == 'object':
        #    self.label_encoder = LabelEncoder()
        #    y_encoded = self.label_encoder.fit_transform(y)
        #    return pd.Series(y_encoded, index=y.index)
        
        # If target needs scaling (regression)
        # elif self.config.get('scale_target', False):
        #     self.target_scaler = StandardScaler()
        #     y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        #     return pd.Series(y_scaled.flatten(), index=y.index)
        
        # If no preprocessing needed
        return y

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """Preprocess the data and split into train/test sets"""
        #from sklearn.model_selection import train_test_split
        
        

        # Separate features and target
        target_col = self.config['data']['target_column']

        # drop target which is null, maybe can have a method to do it.
        missing_targets = data[data[target_col].isna()]
        print(missing_targets)
        data = data.dropna(subset=[target_col])

        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Preprocess target variable if needed
        y = self.preprocess_target(y)
        print(y.dtype)

        # Create and fit preprocessor
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()
        
        #X_train_processed = self.preprocessor.fit_transform(data)
        #print(X_train_processed.isna().sum())
        #print(X_train_processed)
        #print(X_train_processed.dtypes)
        #print(X_train_processed[X_train_processed["num_adults"] == "one"])
        #print(X_train_processed[X_train_processed["room"].isna()])
        
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)


        
        self.logger.info(f"Data preprocessing completed. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor
