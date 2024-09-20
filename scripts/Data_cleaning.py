import pandas as pd
import numpy as np

def preprocess_data(df):
    
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    
    df["Season"] = np.where(df["Month"].isin([3, 4, 5]), "Spring",
                            np.where(df["Month"].isin([6, 7, 8]), "Summer",
                                     np.where(df["Month"].isin([9, 10, 11]), "Autumn",
                                              np.where(df["Month"].isin([12, 1, 2]), "Winter", "None"))))

   
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])

    
    df[["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]] = df[["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]].fillna(0)

    
    df[["Store", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]] = df[["Store", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]].fillna(0)

   
    return df



