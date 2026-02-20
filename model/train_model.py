from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.march_madness_data import MarchMadnessData
from src.model import tts, rscv
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import joblib


UPSAMPLE = 2  # upsample extreme data. Sets tss param
TARGET = 'Target_Score'  # our target feature

# --- Load data and preprocess ---
mm = MarchMadnessData().load().transform().collect()

# Split data into test and train
X_train, X_test, y_train, y_test = tts(mm, TARGET, 0.85, upsample=UPSAMPLE)

# For use later, full X and y
X = mm.drop(TARGET)
y = mm.select(TARGET)

cv_search = rscv(X_train.shape[1])
cv_search.fit(X_train, y_train)

# Best hyperparameters
print('Hyperparams:', cv_search.best_params_)

# Retrain on full training set
best_estimator = cv_search.best_estimator_
best_estimator.fit(X_train, y_train)

# Predict spreads
y_pred = best_estimator.predict(X_test)

# --- Evaluate ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print('SVR MSE: %.2f' % mse)
print('SVR RMSE: %.2f' % rmse)
print('SVR MAE: %.2f' % mae)

z = pl.concat([
    pl.DataFrame(y_pred, schema=['Pred_Score']),
    pl.DataFrame(y_test, schema=['Score'])
], how='horizontal')

# Predict Scores
y_train_pred = best_estimator.predict(X_train)
z_train = pl.concat([
    pl.DataFrame(y_train_pred, schema=['Pred_Score']),
    pl.DataFrame(y_train, schema=['Score'])
], how='horizontal')

# Plot: True vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(z['Score'], z['Pred_Score'], alpha=0.8, label='Test')
plt.scatter(z_train['Score'], z_train['Pred_Score'], alpha=0.05, marker='o', label='Train')
plt.axhline(z['Score'].mean(), color='green', linestyle=':', label='Naive Mean')
plt.plot([z['Score'].min() - 20, z['Score'].max() + 20], [z['Score'].min() - 20, z['Score'].max() + 20],
         alpha=0.4, color='k', linestyle='-', label='Perfect Prediction')
plt.plot([z['Score'].min(), z['Score'].max()], [z['Score'].min() + 5, z['Score'].max() + 5],
         alpha=0.4, color='r', linestyle='--', label='Score +-5')
plt.plot([z['Score'].min(), z['Score'].max()], [z['Score'].min() - 5, z['Score'].max() - 5],
         alpha=0.4, color='r', linestyle='--')
plt.xlabel('True Score')
plt.ylabel('Predicted Score')
plt.title('True vs Predicted Score')
plt.ylim(20, 120)
plt.xlim(20, 120)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'model/model1')

# Map importance to feature names
importance_df = (
    pl.DataFrame({'Feature': best_estimator.named_steps.model.feature_names_in_,
                  'Importance': best_estimator.named_steps.model.feature_importances_})
    .filter(pl.col('Importance') > 0)
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
    'search_params': cv_search.best_params_,
    'upsample': UPSAMPLE,
}

# Save model pipeline and metadata
joblib.dump({'pipeline': best_estimator,
             'metadata': metadata},
            f'model/model1.joblib')

# # Example using the saved pipeline -----------------------------------
#
# # Load the saved pipeline
# saved_model = joblib.load('model/model.joblib')
#
# # Get the pipeline
# loaded_pipeline = saved_model['pipeline']
# loaded_transformer = saved_model['y_transformer']
#
# # Get the metadata
# metadata = saved_model['metadata']
#
# # Print metadata
# print('Mean Absolute Error:', metadata['mean_absolute_error'])
# print('Mean Squared Error:', metadata['mean_squared_error'])
# print('Root Mean Squared Error:', metadata['root_mean_squared_error'])
# print('Feature Names:', metadata['feature_names'][0:5], '...')
# print('Feature Importance:', metadata['feature_importance'][0:5], '...')
# print('CV Search Parameters:', metadata['search_params'])
# print('Upsample:', metadata['upsample'])
#
# # Predict on new data
# # Where `z` is new data
# predictions = loaded_pipeline.predict(X)
# loaded_transformer.inverse_transform(predictions.reshape(-1, 1)).ravel()
