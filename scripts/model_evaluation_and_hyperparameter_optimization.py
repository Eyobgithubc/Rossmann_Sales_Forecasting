from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def evaluate_random_forest_with_cv1(processed_df, target_column='Sales'):
    X_train, X_test, y_train, y_test = prepare_data(processed_df, target_column)
    
    # Define the RandomForestRegressor
    rf_regressor = RandomForestRegressor(random_state=0)
    
    # Hyperparameter grid with fewer options
    param_grid = {
        'n_estimators': [10],  
        'max_depth': [10],     
        'min_samples_split': [2], 
        'min_samples_leaf': [1]   
    }
    
    # Use GridSearchCV with fewer cross-validation folds and parallel processing
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Perform cross-validation with the best model using fewer folds
    cv_results = cross_validate(best_model, X_train, y_train, cv=2, scoring=['r2', 'neg_mean_squared_error'])
    
    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = -cv_results["test_neg_mean_squared_error"].mean()
    rmse = mse ** 0.5

    # Store results
    results = {
        "Mean Squared Error": round(mse, 2),
        "Root Mean Squared Error": round(rmse, 2),
        "R^2 Score": round(cv_results["test_r2"].mean(), 2),
        "Best Parameters": best_params
    }
    
    # Return both results and the best model
    return results, best_model
