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
