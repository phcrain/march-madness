import polars as pl
from scipy.stats import randint, loguniform
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureOrder(BaseEstimator, TransformerMixin):
    """Reorder features into pipleline's expected order"""
    def __init__(self, feature_order=None):
        self.feature_order = feature_order

    def fit(self, X, y=None):
        # Set feature_order, if not provided
        if self.feature_order is None:
            self.feature_order = X.columns.tolist()
        return self

    def transform(self, X):
        # Ensure all expected columns are present
        if set(self.feature_order) - set(X.columns):
            raise ValueError(f"Missing columns in input: {set(self.feature_order) - set(X.columns)}")
        return X[self.feature_order]  # return ordered features


# Split data into test and train
def tts(df: pl.DataFrame, target: str, train_size: float = 0.75, how: str = 'higher', upsample: int = 0):
    """Split data into train and test sets by year

    Parameters
    ----------
    df: pl.DataFrame
        Prepared data
    target: str
        Target field
    train_size: float, default = 0.75
        Float falling between 0-1. The proportion of data the training set should make up.
        Test data will be made of the compliment of this value (1 - train_size)
    how: {'higher', 'lower', 'closest'}, default = 'higher'
        When the exact proportion of `train_size` cannot be achieved, how should the year to filter on be
        chosen. If higher, the greater year is chosen, meaning more training data than specified in `train_size`
        will be provided (and less test data). If lower, the opposite is true. If closest, the year giving the
        closest proportion to `train_size` will be used. Ties between years with `how` set to closest will
        result in the higher year being selected to split.
    upsample: int, default = 0
        Int indicating how many times to upsample extreme values in the training split. If set to >1, extreme
        values will be scaled (e.g., doubled [2], tripled [3]) in each year of the training set during the split.
    """
    assert how in {'higher', 'lower', 'closest'}, '`how` must be one of {"higher", "lower", "closest"}'
    assert 0 < train_size < 1, '`split` must be a float between 0 and 1 (non-inclusive)'

    # Get sorted list of unique years
    years = df['Year'].value_counts().sort('Year')
    n = sum(years['count'])

    upper_year = None
    lower_year = None
    best_diff = (None, 0)
    total_count = 0
    for year, count in years.iter_rows():
        total_count += count
        fract = total_count / n
        if fract < train_size:
            lower_year = year
            best_diff = (year, train_size - fract)
        elif fract == train_size:
            lower_year = year
            upper_year = year
            best_diff = (year, 0)
        else:
            upper_year = year
            if (fract - train_size) <= best_diff[1]:
                best_diff = (year, fract - train_size)
            break

    if how == 'higher':
        split_year = upper_year
    elif how == 'lower':
        split_year = lower_year
    else:
        split_year = best_diff[0]

    train = df.filter(pl.col('Year').le(split_year))
    test = df.filter(pl.col('Year').gt(split_year)).drop('Year')
    if upsample > 1:
        def __upsampler(group, var=target):
            spread = group[var]
            high_thresh = spread.quantile(0.9)
            low_thresh = spread.quantile(0.1)
            # Select extremes and non-extremes
            extremes = group.filter(pl.col(var).ge(high_thresh) | pl.col(var).le(low_thresh))
            non_extremes = group.filter(pl.col(var).lt(high_thresh) & pl.col(var).gt(low_thresh))
            upsampled_extremes = pl.concat([extremes] * upsample)
            # Concatenate without shuffling
            return pl.concat([non_extremes, upsampled_extremes])
        train = train.group_by('Year').map_groups(__upsampler)
    train = train.drop('Year')

    return (
        train.drop(target).to_pandas(),
        test.drop(target).to_pandas(),
        train.select(target).to_numpy().ravel(),
        test.select(target).to_numpy().ravel()
    )


def mae_scorer():
    # Use a regression scoring function
    return make_scorer(mean_absolute_error, greater_is_better=False)


def rscv(num_feats):
    # Init a time series cross-validator
    tss = TimeSeriesSplit(n_splits=3)

    # Init the regressor
    model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=99,
        n_jobs=-1
    )

    # Define the pipeline
    grid_pipeline = Pipeline([
        ('order', FeatureOrder()),
        ('model', model)
    ])

    # --- Train xgb with hyperparameter tuning ---
    param_dist = {
        'model__n_estimators': randint(600, 1800),
        'model__max_depth': randint(3, 5),
        'model__learning_rate': loguniform(0.01, 0.3),
        'model__min_child_weight': randint(1, 6),
        'model__gamma': loguniform(1e-8, 10),
        'model__subsample': loguniform(0.4, 0.9),
        'model__colsample_bytree': loguniform(0.6, 1.0),
        'model__reg_lambda': loguniform(0.01, 7),
    }

    return RandomizedSearchCV(grid_pipeline, param_dist, cv=tss, scoring=mae_scorer(), verbose=1, n_jobs=-1, n_iter=200)
