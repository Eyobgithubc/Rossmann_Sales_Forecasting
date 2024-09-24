<<<<<<< HEAD
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



=======
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor

def prepare_data(processed_df, target_column='Sales', sample_size=None):
    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]
    if sample_size is not None:
        X = X.sample(n=sample_size, random_state=42)
        y = y.loc[X.index]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_models_with_cv(processed_df, target_column='Sales', sample_size=None):
    X_train, X_test, y_train, y_test = prepare_data(processed_df, target_column, sample_size)

    models = [
        "DecisionTreeRegressor",
        "BaggingRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor"
    ]
    
    regressions = [
        DecisionTreeRegressor(random_state=0),
        BaggingRegressor(random_state=0, n_jobs=-1),  # Use parallel processing
        RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=-1),  # Use fewer trees and parallel processing
        GradientBoostingRegressor(random_state=0)
    ]

    results = {}
    rf_model = None  # Initialize variable to store the RandomForestRegressor

    for model_name, regressor in zip(models, regressions):
        # Fit the model with parallel processing if applicable
        regressor.fit(X_train, y_train)  
        
        # Cross-validate with parallel processing
        cv_results = cross_validate(regressor, X_train, y_train, cv=3, scoring=['r2', 'neg_mean_squared_error'], n_jobs=-1)
        
        mse = -cv_results["test_neg_mean_squared_error"].mean()
        rmse = mse ** 0.5
        
        results[model_name] = {
            "Mean Squared Error": round(mse, 2),
            "Root Mean Squared Error": round(rmse, 2),
            "R^2 Score": round(cv_results["test_r2"].mean(), 2)
        }

        if model_name == "RandomForestRegressor":
            rf_model = regressor  # Store the fitted RandomForestRegressor

    return results, rf_model  # Return both results and the fitted RandomForestRegressor
>>>>>>> task-2
