import pandas as pd
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
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
=======

def preprocess_data(df):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract new features
    df['Weekday'] = df['Date'].dt.weekday
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['IsBeginningOfMonth'] = (df['Day'] <= 10).astype(int)
    df['IsMidMonth'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
    df['IsEndOfMonth'] = (df['Day'] > 20).astype(int)
>>>>>>> task-2

    # Handle NaN values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(-1)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(-1)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(-1)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(-1)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')

    # Convert categorical columns to numeric
    categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop irrelevant columns
    df.drop(columns=['Date'], inplace=True)

    # List of numerical columns to scale
    numerical_cols = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                      'Promo2SinceWeek', 'Promo2SinceYear', 'Month', 'Year', 'Day']

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the numerical columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
