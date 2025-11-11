import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.helpers import setup_logging
import logging
from sklearn.preprocessing import FunctionTransformer
from data.transformers.lowercase_transformer import LowercaseTransformer
from data.transformers.int_transformer import IntTransformer
from data.transformers.checkout_day_transformer import CheckoutDayTransformer
from data.transformers.num_adults_transformer import NumAdultsTransformer
from data.transformers.first_time_transformer import FirstTimeTransformer
from data.transformers.month_transformer import MonthTransformer
from data.transformers.price_transformer import PriceTransformer
from data.imputers.price_imputer import PriceImputer
#from data.imputers.modeRoomImputer import ModeRoomImputer
from data.imputers.room_imputer import RoomImputer
from data.transformers.has_children_transformer import HasChildrenTransformer
from data.transformers.stayed_days_transformer import StayedDaysTransformer
from data.transformers.log1p_transformer import Log1pTransformer
from data.encoders.cyclical_encoder import CyclicalEncoder
from data.transformers.drop_columns_transformer import DropColumnsTransformer
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.data_preprocessor = None

        # Init logging
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def validate_config(self, features: list ) -> bool:
        """Validate all feature name is not None"""

        for feature in features:
            if feature is None:
                return False
        return True
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline based on configuration"""

        config_column_mappings = self.config['preprocessing']['column_mappings']

        booking_id = config_column_mappings['identifier']['booking_id']
        no_show = config_column_mappings['target']['no_show']
        branch = config_column_mappings['categorical']['branch']
        booking_month = config_column_mappings['temporal']['booking_month']
        arrival_month = config_column_mappings['temporal']['arrival_month']
        arrival_day = config_column_mappings['temporal']['arrival_day']
        checkout_month = config_column_mappings['temporal']['checkout_month']
        checkout_day = config_column_mappings['temporal']['checkout_day']
        country = config_column_mappings['categorical']['country']
        first_time = config_column_mappings['boolean']['first_time']
        room = config_column_mappings['categorical']['room']
        price = config_column_mappings['numerical']['price']
        platform = config_column_mappings['categorical']['platform']
        num_adults = config_column_mappings['numerical']['num_adults']
        num_children = config_column_mappings['numerical']['num_children']

        impute_room_strategy = self.config['preprocessing']['impute_room_strategy']
        impute_price_strategy = self.config['preprocessing']['impute_price_strategy']
        
        all_features = [
            booking_id,
            no_show,
            branch,
            booking_month,
            arrival_month,
            checkout_month,
            checkout_day,
            country,
            first_time,
            room,
            price,
            platform,
            num_adults,
            num_children
        ]

        if not self.validate_config(all_features):
            raise ValueError("Missing feature name in config")
    
        lowercase_features = [branch, booking_month, arrival_month, checkout_month, country, first_time, num_adults, platform, room, price]
        int_features = [arrival_day, num_children, checkout_day]#, num_adults, first_time]

        cyclical_features = {
            arrival_month: 12,
            checkout_month: 12,
            booking_month: 12,
            arrival_day: 31,
            checkout_day: 31
        }
        
        drop_features = [booking_id]
        month_features = [booking_month, arrival_month, checkout_month]
        robust_features = [price]
        minmax_features = [num_adults]#, num_children]
        one_hot_features = [branch, country, room, platform]
        depend_features = [room, branch]

        drop_features2 = [arrival_month, checkout_month, arrival_day, checkout_day, num_children]

        feature_transformer = ColumnTransformer([
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_features),
            ('robust', RobustScaler(), robust_features),
            ('minmax', MinMaxScaler(), minmax_features), 
            ], remainder='passthrough')

        # Create the full pipeline
        preprocessor = Pipeline([
            ('drop_cols', DropColumnsTransformer(drop_features)),
            ('lowercase', LowercaseTransformer(columns=lowercase_features)),
            ('clean_num_adults', NumAdultsTransformer(column_name=num_adults)),
            ('clean_first_time', FirstTimeTransformer(column_name=first_time)),
            ('clean_month', MonthTransformer(columns=month_features)),
            ('clean_price', PriceTransformer(column_name=price)),
            ('impute_price', PriceImputer(
                column_name=price,
                depend_features=depend_features,
                strategy=impute_price_strategy
            )),
            ('impute_room', RoomImputer(
                column_name=room,
                price_column=price,
                branch_column=branch,
                strategy=impute_room_strategy
            )),
            ('log_price', Log1pTransformer(columns=[price])),
            ('int_convert', IntTransformer(columns=int_features)),
            ('clean_checkout_day', CheckoutDayTransformer(column_name=checkout_day)),
            ('add_has_children', HasChildrenTransformer(column_name=num_children)),
            ('add_stayed_days', StayedDaysTransformer(
                column_arrival_month=arrival_month,
                column_arrival_day=arrival_day,
                column_checkout_month=checkout_month,
                column_checkout_day=checkout_day
            )),
            ('cyclical_encode', CyclicalEncoder(columns_period_map=cyclical_features)),
            ('drop_cols2', DropColumnsTransformer(drop_features2)),
            ('encode_features', feature_transformer)
        ])

        self.preprocessor = preprocessor
        #return preprocessor
    

    def preprocess_target(self, y: pd.Series) -> pd.Series:
        """Preprocess the target variable
            convert the target dtype from float to int
        """
        if y.dtype == "float64":
            y = y.astype(int)
        return y

    def preprocess_data(self, data_df: pd.DataFrame) -> tuple:
        """Preprocess the data and split into train/test sets
            1) identify the target column (no_show)
            2) check any target value is null, investigate the null target rows, drop the row or impute?
            3) separate data into X features and y target 
            4) preprocess the target - convert from float to int
            5) 
        """

        no_show = self.config['preprocessing']['column_mappings']['target']['no_show']

        # drop target which is null, maybe can have a method to do it.
        #missing_targets = data[data[no_show].isna()]
        
        # Drop the row with null target 
        data_df = data_df.dropna(subset=[no_show])

        X = data_df.drop(columns=[no_show])
        y = data_df[no_show]
        
        # Preprocess target variable if needed
        y = self.preprocess_target(y)
        
        # Create and fit preprocessor
        #if self.preprocessor is None:
        #    self.create_preprocessing_pipeline()
        self.create_preprocessing_pipeline()
        
    
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['preprocessing']['test_size'],
            random_state=self.config['preprocessing']['random_state'],
            stratify=y
        )
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        final_columns = self.preprocessor.get_feature_names_out()
        print(final_columns)

        #X_train_processed = preprocessor.fit_transform(X_train)
        X_train_processed = pd.DataFrame(
            X_train_processed, 
            columns=self.preprocessor.get_feature_names_out(), 
            index=X_train.index
        )
    
        X_test_processed = pd.DataFrame(
            X_test_processed, 
            columns=self.preprocessor.get_feature_names_out(), 
            index=X_test.index
        )

        
        self.logger.info(f"Data preprocessing completed. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor
