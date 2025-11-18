# a. Personal Details
Name: Sian Soon Yee
Email: soonyee80@gmail.com

# b. Overview of the submitted folder and the folder structure.

# Folder Structure

```
./
|── eda.ipynb
|── pytest.ini
|── requirements.txt
|── run.sh
|── src/
      |── config.yaml
      |── data/
            |── data_loader.py
            |── data_preprocessor.py
            |── db.py
            |── encoders/
            |      |── cyclical_encoder.py
            |
            |── imputers/
                  |── price_imputer.py
                  |──room_imputer.py
    - transformers
      - checkout_day_transformer.py
      - drop_columns_transformer.py
      - drop_na_transformer.py
      - first_time_transformer.py
      - has_children_transformer.py
      - int_transformer.py
      - log1p_transformer.py
      - lowercase_transformer.py
      - month_transformer.py
      - num_adults_transformer.py
      - price_transformer.py
      - stayed_days_transformer.py
      - zero_to_nan_transformer.py
  - exception
    - config_key_error.py
  - main.py
  - mlpipeline.py
  - models
    - feature_selector.py
    - model_evaluator.py
    - model_trainer.py
  - tests
    - test_checkout_day_transformer.py
    - test_cyclical_encoder.py
    - test_drop_columns_transformer.py
    - test_hello.py
    - test_log1p_transformer.py
  - utils
    - helper.py
    - model_helper.py
```

# c. Instructions for executing the pipeline and modifying any parameters.

## Running the Pipeline
Run the bash script by typing 
```
./run.sh 

```
in a bash shell, the script required excutable permission.

The script will create a Python virtual environment, and install all the dependencies in the requirements.txt, finally will start the pipeline in the main.py.

## Modifying Behavior
We can modify the config.yaml in the ./src directory to allow easy experimentation of different algorithms and
parameters, just modify params in the config.yaml, and run the ./run.sh again for the new settings.

config.yaml:

- data:
  - db_path: "data/noshow.db"
  - table_name: "noshow"
  - required_columns:
    - The columns name in table must be matching required_columns
   
- preprocessing:
  - test_size: 0.2
  - random_state: 42
  - impute_room_strategy: "median"
  - impute_price_strategy: "median"
  - column_mappings:                  (map features to group type)
    - identifier:
      - booking_id: "booking_id"
    - target:
      - no_show: "no_show"
    - numerical:
      - price: "price"
      - num_adults: "num_adults" 
      - num_children: "num_children"
    - categorical:
      - branch: "branch"
      - room: "room"
      - country: "country"
      - platform: "platform"
    - temporal:
      - arrival_month: "arrival_month"
      - checkout_month: "checkout_month"
      - booking_month: "booking_month"
      - arrival_day: "arrival_day"
      - checkout_day: "checkout_day"
    - boolean:
      - first_time: "first_time"

- training:
  - cv_folds: 5                                          (set cv folds to use)
  - scoring_metric: "f1"                                 (set scoring metric to use)
  - save_model: True                                     (whether to save model)
  - model_output_path: "models/trained_model.pkl"        (path to save model)
  - tuning: False                                        (False=no tuning, True=tune then train model with tuned params)
  - select_feature: False                                (False=use all features, True=use selectFromModel() features)
  - drop_feature: True                                   (drop features according to feature_selection -> drop_features)
  - cross_validate: False                                (True=perform cross validation during training, False=skip cross validation)

- feature_selection:
  - feature_selection_threshold: "mean"           ("median", "mean")
  - features_to_drop:                             (select features to drop, only happens if training -> drop_feature: True)
    - features...                                 (check the features available to drop in the command output after preprocessing step)

- tuning:                                (tuning only happens if training -> tuning: True)
  - search_strategy: "random"            ("grid", "random")
  - n_iter: 20
  - n_jobs: -1
  - random_state: 42
  - hyperparameters:                      (Hyperparameters parameter grid for tuning)
    - random_forest:
      - params grid ...
    - logistic_regression:
      - params grid ...
    - xgboost:
      - params grid ...

- model:                           (select model for training, tuning)
  - algorithm: "random_forest"     (model selection: "random_forest", "logistic_regression", "xgboost")
  - hyperparameters:
    - random_forest:
      - params ...
    - logistic_regression:
      - params ...
    - xgboost:
      - params ...
     
- evaluation:                   (evaluation settings)
  - save_reports: true          (save report if true)
  - reports_path: "reports/"    (save location)
  - metrics:                    (metrics that will be calculated)  
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "f1_macro"
    - "roc_auc"

## Description of logical steps/flow of the pipeline. If you find it useful, please feel free to include suitable visualization aids (eg, flow charts) within the README.

### Flow chart

```
        ┌───────────────┐
        │ Load Data     │
        └───────┬───────┘
                │
  ┌─────────────▼─────────────┐
  │ Split into Train/Test set │
  └─────────────┬─────────────┘
                |
        ┌───────▼──────────────┐
        │ Preprocessing        │
        │ - Cleaning           |
        │ - Transform          |
        │ - Imputation         │
        │ - Encoding           |
        │ - Feature Engineering|
        └───────┬──────────────┘
                │
  ┌─────────────▼─────────────┐
  │ Optional Feature Dropping │
  └─────────────┬─────────────┘
                │
  ┌─────────────▼─────────────┐
  │ Optional Feature Selection│
  └─────────────┬─────────────┘
                │
    ┌───────────▼────────────┐
    │ Optional Model Tuning  │
    └───────────┬────────────┘
                │
     ┌──────────▼─────────┐
     │ Model Training     │
     └──────────┬─────────┘
                │
     ┌──────────▼─────────┐
     │ Evaluation         │
     └──────────┬─────────┘
                │
     ┌──────────▼─────────┐
     │ Save Model & Report│
     └────────────────────┘
```

### Detailed Pipeline Steps:

### Data Loading
Loads SQLite table and validates required columns.

---

### Split data
Split data into training and test sets.

---

### Preprocessing
Operates on raw dataframe in a pipeline:

- Drop unused columns  
- Convert text to lowercase  
- Convert integers  
- Clean month/day fields  
- Price transformation + log1p  
- Imputation for price & room  
- Add engineered features:
  - has_children  
  - stayed_days  
- Cyclical encoding  
- Target encoding  
- ColumnTransformer scaling & one-hot encoding  

---

### Feature Dropping (Optional)
Look into shell output after preprocessing, which are the final features available, select those that you to drop to experiment with the training.

If enabled:
```
training -> drop_feature: true
```

Drops columns defined under:
```
feature_selection -> features_to_drop
```

---

## 4. Feature Selection (Optional)

If enabled:
```
training -> select_feature: true
```

Performs SelectFromModel based on model importance.

---

## 5. Hyperparameter Tuning (Optional)

If enabled:
```
training -> tuning: true
```

Runs:
- GridSearchCV OR RandomizedSearchCV  
- Uses metric defined under:

```
training -> scoring_metric
```

Returning **best_params**.

---

## 6. Model Training

Trains final model using:
- Selected model class  
- Best params (if tuning enabled)  
- CV scoring if enabled  

---

## 7. Evaluation

Computes:
- Accuracy  
- Precision  
- Recall  
- F1  
- F1 Macro  
- ROC-AUC  
- Confusion Matrix  
- Classification Report  

Saves reports under `/reports/`.

---

## 8. Saving the Model

Saved object includes:
- Preprocessor  
- Feature drop list  
- Feature selector  
- Trained model  

Saved under:
```
models/trained_model.pkl
```

---

## Quick summary of EDA

Issues found during EDA, include:
- A row with null target - no_show
- Missing values in room and price features
- Incorrect data type and mixed currency for price feature
- Some features with incorrect data types
- Negative checkout_day
- Mixed case for arrival_month (88 unique months)

Please see next sections on what and how the features are proccessed.

Strong predictor features:
- branch
- country
- first_time


## Describe how the features in the dataset are processed

## Data Preprocessing Summary

| Seq | Feature         | Preprocess Method                                              | Example From                | To / Result                  | Outcome / Explanation                                                                 | Notes |
|-----|-----------------|----------------------------------------------------------------|-----------------------------|------------------------------|----------------------------------------------------------------------------------------|-------|
| 1   | no_show         | Drop rows with null target                                     | —                           | row dropped                  | Ensures target is valid                                                               | Only 1 null row |
| 2   | booking_id      | Drop feature                                                   | —                           | removed                      | Identifier not useful for modeling                                                    | Not predictive |
| 3   | branch          | Lowercase strings                                              | Changi                      | changi                       | Standardizes category text                                                             | Consistency |
| 3   | booking_month   | Lowercase strings                                              | December                    | december                     | Standardizes text before mapping to integers                                          | Consistency |
| 3   | arrival_month   | Lowercase strings                                              | December                    | december                     | Same as above                                                                          | |
| 3   | checkout_month  | Lowercase strings                                              | December                    | december                     | Same as above                                                                          | |
| 3   | country         | Lowercase strings                                              | China                       | china                        | Standardizes country names                                                             | |
| 3   | first_time      | Lowercase strings                                              | Yes                         | yes                          | Standardizes values                                                                    | |
| 3   | num_adults      | Lowercase strings (if object)                                 | One                         | one                          | Fix inconsistent text formats                                                          | |
| 3   | platform        | Lowercase strings                                              | Email                       | email                        | Standardizes values                                                                    | |
| 3   | room            | Lowercase strings                                              | King                        | king                         | Standardizes values                                                                    | |
| 3   | price           | Lowercase strings                                              | USD199.99                   | usd199.99                    | Prepares for numeric conversion                                                        | |
| 4   | num_adults      | Convert object → int64                                         | “one”, “1”                  | 1                            | Fix invalid dtype                                                                      | Required |
| 5   | first_time      | Binary conversion (yes/no → 1/0)                               | yes, no                     | 1, 0                         | Creates accurate binary feature                                                        | |
| 6   | booking_month   | Map month text → month index (1–12)                           | january                     | 1                            | Prepares for cyclical encoding                                                         | |
| 6   | arrival_month   | Map month text → month index                                   | february                    | 2                            | Same                                                                                   | |
| 6   | checkout_month  | Map month text → month index                                   | january                     | 1                            | Same                                                                                   | |
| 7   | price           | Convert object → float                                         | USD100.00                   | 100.0                        | Fix dtype                                                                              | Required |
| 8   | price           | Impute None using nearest neighbor (branch, price group)       | None                        | 456                          | Handles missing price values                                                           | Many missing |
| 9   | room            | Impute None using nearest neighbor (price, branch group)       | None                        | king                         | Fills missing room types                                                               | Many missing |
| 10  | price           | log1p transform                                                | 2000                        | 7.6                          | Compresses extreme outliers                                                            | Good for RF/LR |
| 11  | arrival_day     | float → int64                                                  | 25.0                        | 25                           | Fix dtype                                                                              | |
| 11  | num_children    | float → int64                                                  | 1.0                         | 1                            | Fix dtype                                                                              | |
| 11  | checkout_day    | float → int64                                                  | 22                          | 22                           | Fix dtype                                                                              | |
| 12  | checkout_day    | Fix negative values                                            | -5                          | 5                            | Corrects invalid dates                                                                 | |
| 13  | has_children    | New feature: num_children > 0                                  | 0,1,2                       | 0/1                          | Helps model capture child/no-child behavior                                            | Useful |
| 14  | stayed_days     | checkout_date – arrival_date                                   | Aug 12 – Aug 10             | 2                            | Length-of-stay feature                                                                 | Highly predictive |
| 15  | arrival_month   | Cyclical encoding (12 months)                                  | 1/12                        | sin, cos                     | Represents seasonality                                                                 | Removes ordinality |
| 15  | checkout_month  | Cyclical encoding                                              | 1/12                        | sin, cos                     | Same                                                                                   | |
| 15  | booking_month   | Cyclical encoding                                              | 1/12                        | sin, cos                     | Same                                                                                   | |
| 15  | arrival_day     | Cyclical encoding (31 days)                                    | 1/31                        | 0.20, 0.979                  | Represents within-month cycle                                                          | |
| 15  | checkout_day    | Cyclical encoding (31 days)                                    | 1/31                        | 0.20, 0.979                  | Same                                                                                   | |
| 16  | room            | Target Encoding                                                | king                        | 0.28                         | Numeric mean encoding based on no_show rate                                            | Reduces cardinality |
| 16  | country         | Target Encoding                                                | china                       | 0.60                         | Same                                                                                   | |
| 16  | platform        | Target Encoding                                                | email                       | 0.21                         | Same                                                                                   | |
| 17  | branch          | One-hot encoding                                               | changi                      | one_hot__changi              | Creates binary column per branch                                                       | Prevents leakage |
| 17  | price           | Robust scaling                                                 | 1500                        | 1.75                         | Handles outliers better than StandardScaler                                            | Best for long-tailed |
| 17  | num_adults      | MinMax scaling                                                 | 4                           | 1.0                          | Normalizes to 0–1                                                                      | Helps LR/SVM |
| 17  | num_children    | MinMax scaling                                                 | 1                           | 0.5                          | Normalizes to 0–1                                                                      | Helps LR/SVM |


## Explanation of your choice of models

Logistic Regression:
- Target variable is binary, very suitable for Logistic Regression.
- Logistic Regression requires all input features to be numeric. The features are cleaned, converted into numeric form, appropriate scaling such as RobustScaler (price), MinMaxScaler (num_adults, num_children) and Log1p (price), etc, has been done to create a fully numeric feature matrix very suitable for Logistic Regression.
- Logistic Regression performs well as a baseline on tabular data.

Random Forest:
- Handle well the complexities of mixed data types (numeric, categorical), nonlinear relationships, with some engineered features, and outliers, similar to no show dataset.
- Works well with non-linear relationships, e.g extreme price -> lower chance of no-show.
- Robust to outliers and noise, such as the right skewed price feature, extreme high room price.
- Can captures feature interactions automatically e.g room & branch, arrival_month & checkout_month, etc.
- Can provides feature importance, e.g shows which features is important for no_show target, shows whether engineered features are useful (stayed_days, has_children, cyclical sin/cos).
- Random Forest is a tree-based model, these decision trees are not affected by scaling or distribution, whether the data is scaled or not, our data has all been scaled and transformed, thus is suitable for all models to train on it.
- Price feature has extreme outliers for premium rooms, Random Forest splits on thresholds (not distances), so outliers do not affect the model heavily.
- Captures Feature Interactions Automatically

XGBoost:
- It is a gradient boosting algorithm designed specifically for structured tabular data, like no show dataset.
- Can handle complex patterns and subtle relationships, suitable for mixed categorical, numerical data, engineered features, and nonlinear relationships.
- Can learns very fine-grained patterns, e.g small variations in percentage of no_show in arrival month/day may help in predictions.
- Works well with Target encoding, e.g captures nonlinearities in target-encoded values well.
- Can handle class imbalance, e.g no-show rate is moderately imbalanced, can improves recall and F1 performance.
- Known to perform better than Logistic Regression.

## Evaluation of the models developed. Any metrics used in the evaluation should also be explained.

Scoring Metric used: F1. 
- It balances precision and recall.
- great for imbalanced datasets like no show dataset.
- Both False Positive(FP) and False Negative(FN) matter in hotel no-shows.
- precision = tp / tp + fp -> how many actual out of predicted no shows.
- recall = tp / tp + fn    -> how many correct out of actual no shows