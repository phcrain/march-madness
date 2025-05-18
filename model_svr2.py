from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from scipy.stats import randint, loguniform
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from math import floor, ceil
from src import march_madness_data
from src.team_stats import get_team_slug
from src import tts
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.inspection import permutation_importance
import joblib

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

# Remove perfectly correlated features:
# Check correlations from all features (remove target)
correlation_matrix = mm.drop('Target_Spread').corr()
col_names = correlation_matrix.columns
correlation_df = (
    pl.LazyFrame({
        'Col1': [col1 for col1 in col_names for col2 in col_names],
        'Col2': [col2 for col1 in col_names for col2 in col_names],
        'Correlation': np.abs(correlation_matrix.to_numpy().flatten())
    })
    .sort('Correlation', descending=True)
    .filter(pl.col('Col1').ne(pl.col('Col2')))  # Remove self correlation
)
# Select one name per pair and remove duplicates
correlation_df = (
    correlation_df
    .filter(pl.col('Correlation').eq(1))
    .with_columns(pl.max_horizontal('Col1', 'Col2').alias('Drop'))
    .select('Drop')
    .unique('Drop')
)
# Get list of columns to drop
perf_corr_cols = correlation_df.collect()['Drop'].to_list()
# Drop cols from mm
mm = mm.drop(perf_corr_cols)

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

# Standardize data
scaler = RobustScaler()

# Init a time series cross-validator
tss = TimeSeriesSplit(n_splits=3)

# Init the regressor
model = SVR(kernel='rbf')

# Define the pipeline
grid_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectKBest(score_func=mutual_info_regression)),
    ('model', model)
])

# --- Train SVR with hyperparameter tuning ---
param_dist = {
    'selector__k': randint(floor(X_train.shape[1] * 0.5), X_train.shape[1]),
    'model__C': loguniform(0.01, 10),
    'model__epsilon': loguniform(0.1, 10),
    'model__gamma': ['scale', 'auto'],
}

# Use a regression scoring function
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

cv_search = RandomizedSearchCV(grid_pipeline, param_dist, cv=tss, scoring=scorer, verbose=1, n_jobs=-1, n_iter=200)
cv_search.fit(X_train, y_train)

# Best hyperparameters
print('Hyperparams:', cv_search.best_params_)

# Retrain on full training set
best_estimator = cv_search.best_estimator_
best_estimator.fit(X_train, y_train_t)

# Predict spreads
y_pred_t = best_estimator.predict(X_test)
y_pred = pt.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()

# --- Evaluate ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("SVR MSE: %.2f" % mse)
print("SVR RMSE: %.2f" % rmse)
print("SVR MAE: %.2f" % mae)

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
plt.savefig(f'model')

# Get model from estimator
best_model = best_estimator.named_steps['model']

# Get feature importance from model
importance_rslt = permutation_importance(best_estimator, X_train, y_train_t, scoring=scorer, n_jobs=-1)
importance_values = importance_rslt.importances_mean

# Extract feature selection from pipeline
selector = best_estimator.named_steps['selector']

# Get features names from X
importance_names = X_train.columns

# Map importance ratios to feature names
importance_df = (
    # Get feature names and importance values
    pl.DataFrame({'Feature': importance_names, 'Importance': importance_values})
    # Remove features that had 0 importance
    .filter(pl.col('Importance') > 0)
    # Get importance as ratio of importance over total importance
    # (where importance eq mean MAE increase with feature removal)
    .with_columns(pl.col('Importance').truediv(pl.col('Importance').sum()).alias('Importance'))
    # Sort feature importance from highest to lowest
    .sort('Importance', descending=True)
)

# Print feature importance
print(importance_df)

# Create metadata dict
metadata = {
    'mean_absolute_error': mae,
    'mean_squared_error': mse,
    'root_mean_squared_error': rmse,
    'feature_names': importance_df['Feature'].to_list(),
    'feature_importance': importance_df['Importance'].to_list(),
    'cvsearch_params': cv_search.best_params_
}

# Save model pipeline and metadata
joblib.dump({'pipeline': best_estimator,
             'y_transformer': pt,
             'metadata': metadata},
            f'model/model.joblib')

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
# print('Mean Absolute Error:', metadata['mean_absolute_error'])
# print('Feature Names:', metadata['feature_names'])
# print('Feature Importance:', metadata['feature_importance'])
# print('CV Search Paramters:', metadata['cvsearch_params'])
#
# # Predict on new data
# # Where `z` is new data
# predictions = loaded_pipeline.predict_proba(X)[:, 1] > threshold