# model_setup.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

def setup_regressors():
    models = [
        "DecisionTreeRegressor",
        "BaggingRegressor",
        "RandomForestRegressor"
    ]
    
    regressors = [
        DecisionTreeRegressor(random_state=0),
        BaggingRegressor(random_state=0),
        RandomForestRegressor(random_state=0, n_jobs=-1)  # Enable parallel processing
    ]
    
    params = {
        models[0]: {
            "min_samples_split": range(2, 10),  # Reduced range
            "max_leaf_nodes": range(2, 10)      # Reduced range
        },
        models[1]: {
            "n_estimators": range(2, 10)         # Reduced range
        },
        models[2]: {
            "max_depth": range(2, 10),            # Reduced range
            "max_features": range(2, 10),         # Reduced range
            "n_estimators": range(2, 10)          # Reduced range
        }
    }
    
    return models, regressors, params


def evaluate_regressors(regressors, models, X, y):
    results = []
    for model_name, regressor in zip(models, regressors):
        try:
            cv_results = cross_validate(
                regressor, 
                X, 
                y, 
                cv=3,  # Reduced number of folds
                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                n_jobs=-1  # Enable parallel processing for cross-validation
            )
            avg_r2 = round(cv_results["test_r2"].mean(), 2)
            avg_rmse = round(-cv_results["test_neg_mean_squared_error"].mean(), 2)  # Negate for RMSE
            avg_mae = round(-cv_results["test_neg_mean_absolute_error"].mean(), 2)  # Negate for MAE
            
            results.append({
                'Model': model_name,
                'R_score': avg_r2,
                'RMSE': avg_rmse,
                'MAE': avg_mae
            })
        
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    return pd.DataFrame(results)



