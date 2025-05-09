from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
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
        pl.col('W_Seed').cast(pl.UInt8),  # convert seed to int
        pl.col('L_Seed').cast(pl.UInt8),  # convert seed to int
        pl.col('Round').map_elements(lambda _: round_values.get(_), pl.UInt8),  # convert names to standardized slugs
        pl.col('W_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
        pl.col('L_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
        pl.lit(np.random.sample(n) > 0.5).alias("tiebreaker"),  # tiebreaker for name A/B assignment
        (pl.col('W_Score') - pl.col('L_Score')).alias('Target_Spread')  # The target value (point spread)
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
    .with_columns((pl.col('A_Score') - pl.col('B_Score')).alias('Raw_Spread'))  # The target value (point spread)
    .with_columns(pl.col('Raw_Spread').sign().mul(pl.col('Raw_Spread').abs().log1p()).alias(
        'Target_Spread'))  # The target value (point spread)
    # .with_columns(pl.col('Raw_Spread').abs().log1p().alias('Target_Spread'))  # The target value (point spread)

    .drop('W_Team', 'L_Team', 'W_Seed', 'L_Seed', 'tiebreaker',
          'W_Score', 'L_Score', 'OT', 'W_Last_Digit', 'L_Last_Digit',
          'A_Score', 'B_Score', 'Raw_Spread')
)

# Read in the team stats
stats = pl.scan_csv('data/season_stats/combined_stats.csv')

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
    .with_columns([pl.col(f'A_{cname}').add(pl.lit(delta)).truediv(pl.col(f'B_{cname}').add(pl.lit(delta)))
                  .alias(f'ratio_{cname}') for cname in cnames])
    .drop([f'A_{cname}' for cname in cnames if cname != 'Seed'])
    .drop([f'B_{cname}' for cname in cnames if cname != 'Seed'])
)

mm = mm.collect()

# Split data into test and train
X_train, X_test, y_train, y_test = tts(mm, 'Target_Spread', 0.85)

# Standardize data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Flatten y for sklearn
y_train_np = y_train.to_numpy().ravel()
y_test_np = y_test.to_numpy().ravel()

# --- Train SVR with hyperparameter tuning ---
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 1, 5],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid.fit(X_train, y_train_np)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Evaluate ---
mse = mean_squared_error(y_test_np, y_pred)
rmse = np.sqrt(mse)

print("SVR MSE: %.2f" % mse)
print("SVR RMSE: %.2f" % rmse)

# --- Compare to naive average and median baselines ---
naive_avg_pred = np.full_like(y_test_np, y_train.mean().item())
naive_med_pred = np.full_like(y_test_np, y_train.median().item())

avg_mse = mean_squared_error(y_test_np, naive_avg_pred)
med_mse = mean_squared_error(y_test_np, naive_med_pred)

print("Naive Average RMSE: %.2f" % np.sqrt(avg_mse))
print("Naive Median RMSE: %.2f" % np.sqrt(med_mse))

import seaborn as sns

# --- Scatter Plot: True vs Predicted Spread ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_np, y=y_pred, alpha=0.6, label="SVR + PCA")
sns.scatterplot(x=y_test_np, y=naive_avg_pred, alpha=0.6, label="Naive Mean", marker="x")
sns.scatterplot(x=y_test_np, y=naive_med_pred, alpha=0.6, label="Naive Median", marker="^")

# Reference line for perfect prediction
min_val = min(y_test_np.min(), y_pred.min())
max_val = max(y_test_np.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

plt.xlabel("True Spread")
plt.ylabel("Predicted Spread")
plt.title("True vs Predicted Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svr_pca_scatter_output.png")