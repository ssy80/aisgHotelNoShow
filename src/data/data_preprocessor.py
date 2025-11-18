import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.helper import setup_logging, safe_get
import logging
from data.transformers.lowercase_transformer import LowercaseTransformer
from data.transformers.int_transformer import IntTransformer
from data.transformers.checkout_day_transformer import CheckoutDayTransformer
from data.transformers.num_adults_transformer import NumAdultsTransformer
from data.transformers.first_time_transformer import FirstTimeTransformer
from data.transformers.month_transformer import MonthTransformer
from data.transformers.price_transformer import PriceTransformer
from data.imputers.price_imputer import PriceImputer
from data.imputers.room_imputer import RoomImputer
from data.transformers.has_children_transformer import HasChildrenTransformer
from data.transformers.stayed_days_transformer import StayedDaysTransformer
from data.transformers.log1p_transformer import Log1pTransformer
from data.encoders.cyclical_encoder import CyclicalEncoder
from data.transformers.drop_columns_transformer import DropColumnsTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder


class DataPreprocessor:
    """
    Builds and runs the full preprocessing pipeline for the dataset.
    """

    def __init__(self, config: dict):
        """
        Initializes the preprocessor with configuration and logging.
        """
        if config is None:
            raise ValueError("DataPreprocessor __init__: config cannot be None")

        self.config = config
        self.data_preprocessor = None

        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Creates the full preprocessing pipeline:
        - Cleans raw fields
        - Imputes missing values
        - Encodes categorical features
        - Scales numerical features
        - Adds engineered features
        """
        config_column_mappings = safe_get(self.config, 'preprocessing', 'column_mappings', required=True)

        booking_id = safe_get(config_column_mappings, 'identifier', 'booking_id', required=True)
        no_show = safe_get(config_column_mappings, 'target', 'no_show', required=True)
        branch = safe_get(config_column_mappings, 'categorical', 'branch', required=True)
        booking_month = safe_get(config_column_mappings, 'temporal', 'booking_month', required=True)
        arrival_month = safe_get(config_column_mappings, 'temporal', 'arrival_month', required=True)
        arrival_day = safe_get(config_column_mappings, 'temporal', 'arrival_day', required=True)
        checkout_month = safe_get(config_column_mappings, 'temporal', 'checkout_month', required=True)
        checkout_day = safe_get(config_column_mappings, 'temporal', 'checkout_day', required=True)
        country = safe_get(config_column_mappings, 'categorical', 'country', required=True)
        first_time = safe_get(config_column_mappings, 'boolean', 'first_time', required=True)
        room = safe_get(config_column_mappings, 'categorical', 'room', required=True)
        price = safe_get(config_column_mappings, 'numerical', 'price', required=True)
        platform = safe_get(config_column_mappings, 'categorical', 'platform', required=True)
        num_adults = safe_get(config_column_mappings, 'numerical', 'num_adults', required=True)
        num_children = safe_get(config_column_mappings, 'numerical', 'num_children', required=True)

        impute_room_strategy = safe_get(self.config, 'preprocessing', 'impute_room_strategy', required=True)
        impute_price_strategy = safe_get(self.config, 'preprocessing', 'impute_price_strategy', required=True)

        lowercase_features = [branch, booking_month, arrival_month, checkout_month, country, first_time, num_adults, platform, room, price]
        int_features = [arrival_day, num_children, checkout_day]

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
        minmax_features = [num_adults, num_children]
        depend_features = [room, branch]

        one_hot_features = [branch]
        target_encoding_features = [room, country, platform]

        feature_transformer = ColumnTransformer([
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_features),
            ('robust', RobustScaler(), robust_features),
            ('minmax', MinMaxScaler(), minmax_features),
        ], remainder='passthrough')

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
            ('int_convert', IntTransformer(columns=int_features)),
            ('clean_checkout_day', CheckoutDayTransformer(column_name=checkout_day)),
            ('add_has_children', HasChildrenTransformer(column_name=num_children)),
            ('add_stayed_days', StayedDaysTransformer(
                column_arrival_month=arrival_month,
                column_arrival_day=arrival_day,
                column_checkout_month=checkout_month,
                column_checkout_day=checkout_day
            )),
            ('log_features', Log1pTransformer(columns=[price, 'stayed_days'])),
            ('cyclical_encode', CyclicalEncoder(columns_period_map=cyclical_features)),
            ('target_encode', TargetEncoder(cols=target_encoding_features)),
            ('encode_features', feature_transformer)
        ])

        self.preprocessor = preprocessor

    def preprocess_target(self, y: pd.Series) -> pd.Series:
        """
        Converts the target column to int if needed.
        """
        if y.dtype == "float64":
            y = y.astype(int)
        return y

    def preprocess_data(self, data_df: pd.DataFrame) -> tuple:
        """
        Runs the full preprocessing workflow:
        - Drops rows with missing target values
        - Splits into features and target
        - Converts target type
        - Builds preprocessing pipeline
        - Splits into train/test
        - Fits pipeline on train and transforms both
        Returns processed train/test sets and the pipeline.
        """
        no_show = safe_get(self.config, 'preprocessing', 'column_mappings', 'target', 'no_show', required=True)

        data_df = data_df.dropna(subset=[no_show])

        X = data_df.drop(columns=[no_show])
        y = data_df[no_show]

        y = self.preprocess_target(y)

        self.create_preprocessing_pipeline()

        test_size = safe_get(self.config, 'preprocessing', 'test_size', required=True)
        random_state = safe_get(self.config, 'preprocessing', 'random_state', required=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        X_train_processed = self.preprocessor.fit_transform(X_train, y_train) #y_train for target encoder
        X_test_processed = self.preprocessor.transform(X_test)

        X_train_processed = pd.DataFrame(
            X_train_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=X_train.index
        )

        print(X_train_processed['remainder__country'].dtype)
        print(X_train_processed['remainder__room'].dtype)
        print(X_train_processed['remainder__platform'].dtype)

        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=X_test.index
        )

        self.logger.info(
            f"Data preprocessing completed. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}"
        )

        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor
