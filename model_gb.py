from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from src import march_madness_data
from src.team_stats import get_team_slug
from src import tts
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

# --- Load data and preprocess ---

# read in march madness data
mm = march_madness_data()
# row count for march madness dataset
n = mm.select(pl.len()).collect().item()
# mapping to convert ordinal Round into int
round_values = {
    'Round of 64': 1,
    'Round of 32': 2,
    'Sweet 16': 3,
    'Elite 8': 4,
    'Final 4': 5,
    'National Championship': 6
}

mm = (
    mm
    .with_columns(
        pl.col('W_Seed').cast(pl.Int16),  # convert seed to int
        pl.col('L_Seed').cast(pl.Int16),  # convert seed to int
        pl.col('Round').map_elements(lambda _: round_values.get(_), pl.Int16),  # convert names to standardized slugs
        pl.col('W_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
        pl.col('L_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
        pl.lit(np.random.sample(n) > 0.5).alias("tiebreaker"),  # tiebreaker for name A/B assignment
    )
    .with_columns(
        pl.min_horizontal(pl.col('W_Seed'), pl.col('L_Seed')).alias('A_Seed'),  # Assign best seed to A
        pl.max_horizontal(pl.col('W_Seed'), pl.col('L_Seed')).alias('B_Seed'),  # Assign worst seed to B
        pl.when(pl.col('W_Seed') < pl.col('L_Seed')).then(pl.col('W_Team'))  # Assign best seed name to A
        .when(pl.col('W_Seed') > pl.col('L_Seed')).then(pl.col('L_Team'))  # Assign worst seed name to B
        .when(pl.col('tiebreaker')).then(pl.col('W_Team'))  # If seeds are tied, use tiebreaker value
        .otherwise(pl.col('L_Team'))  # If seeds are tied, use tiebreaker value
        .alias('A_Team')
    )
    .with_columns(
        pl.when(pl.col('A_Team').eq(pl.col('W_Team'))).then(pl.col('L_Team'))  # Assign B name based on A name
        .otherwise(pl.col('W_Team')).alias('B_Team')
    )
    .with_columns(
        pl.when(pl.col('A_Team').eq(pl.col('W_Team'))).then(pl.col('W_Score'))  # Assign A score based on A name
        .otherwise(pl.col('L_Score')).alias('A_Score'),
        pl.when(pl.col('A_Team').eq(pl.col('W_Team'))).then(pl.col('L_Score'))  # Assign B score based on A name
        .otherwise(pl.col('W_Score')).alias('B_Score')
    )
    .with_columns((pl.col('A_Score') - pl.col('B_Score')).alias('Target_Spread'))  # The target value (point spread)
    # .with_columns(pl.col('Raw_Spread').sign().mul(pl.col('Raw_Spread').abs().log1p()).alias('Target_Spread'))  # The target value (point spread)
    # .with_columns(pl.col('Raw_Spread').abs().log1p().alias('Target_Spread'))  # The target value (point spread)

    .drop('W_Team', 'L_Team', 'W_Seed', 'L_Seed', 'tiebreaker',
          'W_Score', 'L_Score', 'OT', 'W_Last_Digit', 'L_Last_Digit',
          'A_Score', 'B_Score') #, 'Raw_Spread')
)

# Read in the team stats
stats = pl.scan_parquet('data/combined_stats.parquet')

# Join season stats to march madness data
mm = (
    mm
    .join(stats.rename(lambda cname: "A_" + cname),
          left_on=['A_Team', 'Year'],
          right_on=['A_Team', 'A_Year'],
          how='inner')
    .join(stats.rename(lambda cname: "B_" + cname),
          left_on=['B_Team', 'Year'],
          right_on=['B_Team', 'B_Year'],
          how='inner')
    # Drop team names and year now that we have the stats joined
    .drop('A_Team', 'B_Team')
)

# Feature eng: create diffs for team v opp stats
cnames = mm.collect_schema().names()
for cname in cnames:
    if cname.find('_opp') != -1 and cname.replace('_opp', '') in cnames:
        mm = mm.with_columns(pl.col(cname.replace('_opp', '')).sub(pl.col(cname)).alias(f'{cname}_diff'))
        mm = mm.drop(cname)

# Feature eng:

# Feature eng: create diffs and ratios for team v team matchups
# Get shared A/B team stats
cnames = set([cname[2:] for cname in mm.collect_schema().names()
              if cname.startswith('A_') or cname.startswith('B_')])
cnames.remove('OT')
cnames.remove('OT_std')
cnames.remove('conf_A10')
cnames.remove('conf_ACC')
cnames.remove('conf_Amer')
cnames.remove('conf_B10')
cnames.remove('conf_B12')
cnames.remove('conf_BE')
cnames.remove('conf_MWC')
cnames.remove('conf_P12')
cnames.remove('conf_SEC')
cnames.remove('conf_WCC')
# Offset any 0 values by small delta
delta = 0.000001
# Calculate diffs and ratios
mm = (
    mm
    .with_columns([pl.col(f'A_{cname}').sub(pl.col(f'B_{cname}')).alias(f'diff_{cname}') for cname in cnames])
    #.with_columns([pl.col(f'A_{cname}').add(pl.lit(delta)).truediv(pl.col(f'B_{cname}').add(pl.lit(delta)))
    #              .alias(f'ratio_{cname}') for cname in cnames])
    .with_columns([pl.col(f'A_{cname}').mul(pl.col(f'B_{cname}')).alias(f'prod_{cname}') for cname in cnames])
    .with_columns([pl.col(f'A_{cname}').mul(pl.col(f'A_{cname}')).alias(f'A_power_{cname}') for cname in cnames])
    .with_columns([pl.col(f'B_{cname}').mul(pl.col(f'B_{cname}')).alias(f'B_power_{cname}') for cname in cnames])
    .with_columns([pl.col(f'A_power_{cname}').sub(pl.col(f'B_power_{cname}')).alias(f'diff_power_{cname}')
                   for cname in cnames])
    .drop([f'A_{cname}' for cname in cnames if cname != 'Seed'])
    .drop([f'B_{cname}' for cname in cnames if cname != 'Seed'])
    .drop([f'A_power_{cname}' for cname in cnames])
    .drop([f'B_power_{cname}' for cname in cnames])
)

# Save memory with Float32 instead of 64
mm = mm.with_columns([
    pl.col(col).cast(pl.Float32) for col in mm.collect_schema().names() if mm.collect_schema().get(col) == pl.Float64
])

mm = mm.collect()

#
# corr_mm = mm.corr().to_pandas()
#
# # Select upper triangle of correlation matrix
# upper = corr_mm.where(
#     np.triu(np.ones(corr_mm.shape), k=1).astype(bool)
# )
#
# # Find columns with correlation > threshold
# to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Split data into test and train
X_train, X_test, y_train, y_test = tts(mm, 'Target_Spread', 0.85)
X_train = X_train.to_pandas()
X_test = X_test.to_pandas()
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

# Transform target spread
pt = PowerTransformer(method='yeo-johnson', standardize=False)
y_train_t = pt.fit_transform(y_train.reshape(-1, 1)).ravel()

# For use later, full X and y
X = mm.drop('Target_Spread')
y = mm.select('Target_Spread')

# Init a time series cross-validator
tss = TimeSeriesSplit(n_splits=4)

# Init the regressor
model = XGBRegressor(
    random_state=123,
    n_estimators=1500,
    learning_rate=0.01,
    colsample_bynode=0.5,
    colsample_bytree=0.25,
    max_delta_step=3,
    min_child_weight=10
)

# Define the pipeline
grid_pipeline = Pipeline([
    ('selector', SelectKBest(score_func=mutual_info_regression, k=375)),
    ('model', model)
])

# Define hyperparameter grid (no scale_pos_weight in regression)
param_grid = {
    'model__max_depth': [4, 5, 6],
    'model__alpha': [0.1, 1, 3],
    'model__lambda': [1, 3, 5],
}

# Use a regression scoring function
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Init and fit grid search
grid_search = HalvingGridSearchCV(grid_pipeline, param_grid, cv=tss, scoring=scorer, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train_t)

# Best hyperparameters
print('Hyperparams:', grid_search.best_params_)

# Retrain on full training set
best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train, y_train_t)

# Predict spreads
y_pred_t = best_estimator.predict(X_test)
y_pred = pt.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()

# Evaluate
mse = mean_absolute_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

z = pl.concat([
    pl.DataFrame(y_pred, schema=['Pred_Spread']),
    pl.DataFrame(y_test, schema=['Spread'])
], how='horizontal')

# Predict spreads
y_train_pred_t = best_estimator.predict(X_train)
y_train_pred = pt.inverse_transform(y_train_pred_t.reshape(-1, 1)).ravel()
z_train = pl.concat([
    pl.DataFrame(y_train_pred, schema=['Pred_Spread']),
    pl.DataFrame(y_train, schema=['Spread'])
], how='horizontal')

# Plot: True vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(z['Spread'], z['Pred_Spread'], alpha=0.8, label="Model Prediction")
plt.scatter(z_train['Spread'], z_train['Pred_Spread'], alpha=0.05, marker='o', label="Training Prediction")
plt.scatter(z['Spread'], z.height * [z['Spread'].mean()], alpha=0.6, marker="x", label="Naive Mean")
# plt.scatter(z['Spread'], z.height * [z['Spread'].median()], alpha=0.6, marker="^", label="Naive Median")
plt.plot([z['Spread'].min(), z['Spread'].max()], [z['Spread'].min(), z['Spread'].max()],
         alpha=0.4, color='k', linestyle='--', label="Perfect Prediction")
plt.plot([z['Spread'].min(), z['Spread'].max()], [z['Spread'].min() + 5, z['Spread'].max() + 5],
         alpha=0.4, color='r', linestyle='--', label="Spread +-5")
plt.plot([z['Spread'].min(), z['Spread'].max()], [z['Spread'].min() - 5, z['Spread'].max() - 5],
         alpha=0.4, color='r', linestyle='--')
plt.xlabel("True Spread")
plt.ylabel("Predicted Spread")
plt.title("True vs Predicted Spread")
plt.ylim(-50, 50)
plt.xlim(-50, 50)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output3')


# Get model from estimator
best_model = best_estimator.named_steps['model']

### DEV get feature importance from model
importance = best_estimator.get_booster().get_score(importance_type='weight')
keys = list(importance.keys())
values = list(importance.values())
pl.DataFrame({'feature': keys, 'importance': values}).sort('importance', descending=True)


# Get feature importance from model
importance_values = best_model.feature_importances_

# Extract feature selection from pipeline
selector = best_estimator.named_steps['selector']

# Get features names from X
importance_names = np.array(X_train.columns)[selector.get_support()]

# Map importance ratios to feature names
importance_df = (
    # Get feature names and importance values
    pl.DataFrame({'Feature': importance_names, 'Importance': importance_values})
    # Remove features that had 0 importance
    .filter(pl.col('Importance') > 0)
    # Sort feature importance from highest to lowest
    .sort('Importance', descending=True)
)

# Print feature importance
print(importance_df)

# # Check correlations of selected features:
# correlation_matrix = X.select(importance_df['Feature'].to_list()).corr()
#
# col_names = correlation_matrix.columns
# correlation_df = (
#     pl.DataFrame({
#         'Col1': [col1 for col1 in col_names for col2 in col_names],
#         'Col2': [col2 for col1 in col_names for col2 in col_names],
#         'Correlation': np.abs(correlation_matrix.to_numpy().flatten())
#     })
#     .sort('Correlation', descending=True)
#     .filter(pl.col('Col1').is_in(importance_df['Feature'])  # Keep only rows where both cols are in the selected feats
#             & pl.col('Col2').is_in(importance_df['Feature'])  # Keep only rows where both cols are in the selected feats
#             & (pl.col('Col1') != pl.col('Col2')))  # Remove self correlation
# )
# print(correlation_df)
#
# # Create list of correlation dicts
# corr_list = []
# for field1 in importance_df['Feature'].to_list():
#     corr_dict = {}
#     for field2 in importance_df['Feature'].to_list():
#         if field1 == field2:
#             corr = 1.0
#         else:
#             corr = correlation_df.filter((pl.col('Col1') == field1) & (pl.col('Col2') == field2))['Correlation'][0]
#         corr_dict[field2] = corr
#     corr_list.append(corr_dict)
#
# # Create metadata dict
# metadata = {
#     'classification_report': class_rep,
#     'roc_auc_score': roc_auc,
#     'feature_names': importance_df['Feature'].to_list(),
#     'feature_importance': importance_df['Importance'].to_list(),
#     'feature_correlation': corr_list,
#     'prediction_threshold': threshold
# }
#
# # Save model pipeline and metadata
# joblib.dump({'pipeline': best_estimator,
#              'metadata': metadata},
#             'model/model.joblib')
#
# # Example using the saved pipeline -----------------------------------
#
# # Load the saved pipeline
# saved_data = joblib.load('model/model.joblib')
#
# # Get the pipeline
# loaded_pipeline = saved_data['pipeline']
#
# # Get the metadata
# metadata = saved_data['metadata']
#
# # Print metadata
# print('Classification Report', metadata['classification_report'])
# print('ROC AUC Score:', metadata['roc_auc_score'])
# print('Feature Names:', metadata['feature_names'])
# print('Feature Importance:', metadata['feature_importance'])
# print('Feature Correlation:', pl.DataFrame(metadata['feature_correlation']))
#
# # Get the prediction threshold
# threshold = metadata['prediction_threshold']
#
# # Predict on new data
# # Where `X` is new data
# predictions = loaded_pipeline.predict_proba(X)[:, 1] > threshold
