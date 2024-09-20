import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
def preprocess_data(df, target_column=None, test_size=0.2, random_state=42):
   
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Weekday"] = df["Date"].dt.weekday
    df["Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

   
    df["Season"] = np.where(df["Month"].isin([3, 4, 5]), "Spring",
                            np.where(df["Month"].isin([6, 7, 8]), "Summer",
                                     np.where(df["Month"].isin([9, 10, 11]), "Autumn", "Winter")))

   
    df["MonthPart"] = pd.cut(df["Day"], bins=[0, 10, 20, 31], labels=["Beginning", "Mid", "End"])

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])
    df[["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]] = df[["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]].fillna(0)
    df[["Promo2SinceWeek", "Promo2SinceYear"]] = df[["Promo2SinceWeek", "Promo2SinceYear"]].fillna(0)

 
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

   
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

   
    if 'Date' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['Date'])


    if target_column and target_column in df_encoded.columns:
        target = df_encoded[target_column]
        df_encoded = df_encoded.drop(columns=[target_column])
    else:
        target = None


    numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

  
    non_numeric_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_columns) > 0:
        raise ValueError(f"Non-numeric columns found after preprocessing: {list(non_numeric_columns)}")

 
    if target_column and target is not None:
        df_encoded[target_column] = target


    if target is not None:
        X_train, X_test, y_train, y_test = train_test_split(df_encoded, target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    else:
        return df_encoded


