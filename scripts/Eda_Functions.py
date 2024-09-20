import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from IPython.display import display 

def compare_promo_distribution(train, test):
    logging.info("Compare Promotion Distribution in Train and Test Data.")
    train_promo_dist = train['Promo'].value_counts(normalize=True)
    test_promo_dist = test['Promo'].value_counts(normalize=True)
    print("Promotion Distribution in Train Dataset:")
    print(train_promo_dist)
    print("\nPromotion Distribution in Test Dataset:")
    print(test_promo_dist)
    logging.info("Compare Promotion Distribution in Train and Test Data completed.")

def holiday_sales_analysis(train):
    logging.info("Sales Before, During, and After Holidays.")
    
    holiday_sales = train[train['StateHoliday'] != '0']
    before_holiday_sales = train[train['StateHoliday'] == '0']
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(holiday_sales['Sales'], bins=30, kde=False, color='red', alpha=0.7)
    plt.title('Sales Distribution During Holidays')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(before_holiday_sales['Sales'], bins=30, kde=False, color='blue', alpha=0.7)
    plt.title('Sales Distribution Before Holidays')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    avg_sales_before = before_holiday_sales['Sales'].mean()
    avg_sales_during = holiday_sales['Sales'].mean()
    print(f'Average Sales Before Holidays: {avg_sales_before}')
    print(f'Average Sales During Holidays: {avg_sales_during}')
    
    logging.info("Sales Before, During, and After Holidays Analysis completed.")


def seasonal_trends_analysis(train):
    logging.info("Analyze Seasonal Purchase Behaviors.")
    
    train['Year'] = train['Date'].dt.year
    train['Month'] = train['Date'].dt.month
    monthly_sales = train.groupby('Month')['Sales'].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Month', y='Sales', data=monthly_sales)
    plt.title('Average Sales by Month (Seasonal Trends)')
    plt.show()
    logging.info("Analyze Seasonal Purchase Behaviors completed.")


def sales_customers_correlation(train):
    logging.info("Analyze Correlation Between Sales and Customers.")
    
    corr = train[['Sales', 'Customers']].corr()
    print(f"Correlation Between Sales and Customers: \n{corr}")
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Customers', y='Sales', data=train)
    plt.title('Correlation Between Sales and Number of Customers')
    plt.show()
    logging.info("Analyze Correlation Between Sales and Customers completed.")

def promotions_sales_analysis(train):
    logging.info("Summary of Sales and Customers (with and without promo).")
    promo_summary = train.groupby('Promo').agg({
        'Sales': ['mean', 'median', 'std', 'min', 'max'],
        'Customers': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()

    print("Summary of Sales and Customers (with and without promo):")
    display(promo_summary)  

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=train)
    plt.title('Sales During Promo vs No Promo')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Customers', data=train)
    plt.title('Number of Customers During Promo vs No Promo')
    plt.show()
    logging.info("Summary of Sales and Customers (with and without promo) completed")

def effective_promo_stores(train):
    logging.info("Top 10 Stores for Promo Effectiveness.")
    
    store_promo_sales = train.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
    store_promo_sales['Promo_Effectiveness'] = store_promo_sales[1] - store_promo_sales[0]
    effective_stores = store_promo_sales.sort_values('Promo_Effectiveness', ascending=False).head(10)
    print("Top 10 Stores for Promo Effectiveness:")
    display(effective_stores)
    logging.info("Top 10 Stores for Promo Effectiveness completed")


def open_closed_sales_analysis(train):
    logging.info("Sales Behavior During Store Opening/Closing Times")
    open_sales = train[train['Open'] == 1]['Sales'].mean()
    closed_sales = train[train['Open'] == 0]['Sales'].mean()
    print(f'Average Sales When Stores Are Open: {open_sales}')
    print(f'Average Sales When Stores Are Closed: {closed_sales}')
    logging.info("Sales Behavior During Store Opening/Closing Times completed")


def assortment_sales_analysis(train_store_merged):
    logging.info("Effect of Assortment on Sales")
    assortment_summary = train_store_merged.groupby('Assortment').agg({
        'Sales': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()

    print("Summary of Sales by Assortment Level:")
    display(assortment_summary) 

   
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Assortment', y='Sales', data=train_store_merged)
    plt.title('Sales by Assortment Level')
    plt.show()
    logging.info("Effect of Assortment on Sales completed")

def competitor_distance_sales_analysis(train_store_merged):
    logging.info("Summary of Sales by Competition Distance")
    distance_summary = train_store_merged.groupby(pd.cut(train_store_merged['CompetitionDistance'], bins=5)).agg({
        'Sales': ['mean', 'median', 'std', 'min', 'max'],
        'CompetitionDistance': ['count']
    }).reset_index()

    print("Summary of Sales by Competition Distance:")
    display(distance_summary)  

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=train_store_merged)
    plt.title('Sales vs Competition Distance')
    plt.show()
    logging.info("Summary of Sales by Competition Distance completed")


def competitor_opening_analysis(train_store_merged):
    logging.info("Sales Before and After Competitor Opening")
    competitor_opening_stores = train_store_merged[train_store_merged['CompetitionDistance'].isna()]
    stores_with_new_competitors = train_store_merged[~train_store_merged['CompetitionDistance'].isna()]

    before_competitor_open_sales = competitor_opening_stores['Sales'].mean()
    after_competitor_open_sales = stores_with_new_competitors['Sales'].mean()

    print(f'Average Sales Before Competitor Opening: {before_competitor_open_sales}')
    print(f'Average Sales After Competitor Opening: {after_competitor_open_sales}')
    logging.info("Sales Before and After Competitor Opening completed")
    
def analyze_weekly_sales(data):
    logging.info("Analyzing weekly sales patterns.")
    
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=data, estimator=sum)
    plt.title("Total Sales by Day of the Week")
    plt.ylabel('Total Sales')
    plt.xlabel('Day of the Week')
    plt.show()
    
   
    data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.dayofweek
    

    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['DayOfWeek'] = data['DayOfWeek'].map(day_names)
    
    
    summary_table = data.groupby('DayOfWeek')['Sales'].sum().reset_index()
    summary_table = summary_table.rename(columns={'Sales': 'Total Sales'})
    
 
    summary_table = summary_table.sort_values(by='Total Sales', ascending=True)
    
   
    print("Weekly Sales Summary Table (Sorted):")
    print(summary_table)
    logging.info("Weekly sales analysis completed.")
    
def analyze_store_type_sales(data):
    logging.info("Analyzing the effect of store type on sales.")
    
      
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StoreType', y='Sales', data=data)
    plt.title("Effect of Store Type on Sales")
    plt.xlabel('Store Type')
    plt.ylabel('Sales')
    plt.show()
    
   
    summary_table = data.groupby('StoreType')['Sales'].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
    summary_table = summary_table.rename(columns={
        'sum': 'Total Sales',
        'mean': 'Average Sales',
        'median': 'Median Sales',
        'std': 'Sales Std Dev',
        'count': 'Number of Records'
    })
    
    
    summary_table = summary_table.sort_values(by='Average Sales', ascending=True)
    
   
    print("Sales Summary by Store Type (Sorted by Average Sales):")
    display(summary_table)
    
    logging.info("Store type analysis completed.")

def analyze_holiday_impact(data):
    logging.info("Analyzing the impact of holidays on sales.")
    
    
    state_holiday_summary = data.groupby('StateHoliday')['Sales'].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
    state_holiday_summary = state_holiday_summary.rename(columns={
        'sum': 'Total Sales',
        'mean': 'Average Sales',
        'median': 'Median Sales',
        'std': 'Sales Std Dev',
        'count': 'Number of Records'
    })
    
   
    school_holiday_summary = data.groupby('SchoolHoliday')['Sales'].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
    school_holiday_summary = school_holiday_summary.rename(columns={
        'sum': 'Total Sales',
        'mean': 'Average Sales',
        'median': 'Median Sales',
        'std': 'Sales Std Dev',
        'count': 'Number of Records'
    })
    
   
    state_holiday_summary = state_holiday_summary.sort_values(by='Average Sales', ascending=True)
    school_holiday_summary = school_holiday_summary.sort_values(by='Average Sales', ascending=True)
    
    
    print("Sales Summary by State Holiday (Sorted by Average Sales):")
    print(state_holiday_summary)
    print("\nSales Summary by School Holiday (Sorted by Average Sales):")
    print(school_holiday_summary)
    
 

def analyze_holiday_impact(data):
    
    logging.info("Analyzing the impact of holidays on sales.")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x='Sales', hue='StateHoliday', multiple='stack', bins=30)
    plt.title("Sales Distribution by State Holiday")
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    
  
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x='Sales', hue='SchoolHoliday', multiple='stack', bins=30)
    plt.title("Sales Distribution by School Holiday")
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    
    state_holiday_summary = data.groupby('StateHoliday')['Sales'].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
    state_holiday_summary = state_holiday_summary.rename(columns={
        'sum': 'Total Sales',
        'mean': 'Average Sales',
        'median': 'Median Sales',
        'std': 'Sales Std Dev',
        'count': 'Number of Records'
    })
    
    
    school_holiday_summary = data.groupby('SchoolHoliday')['Sales'].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
    school_holiday_summary = school_holiday_summary.rename(columns={
        'sum': 'Total Sales',
        'mean': 'Average Sales',
        'median': 'Median Sales',
        'std': 'Sales Std Dev',
        'count': 'Number of Records'
    })
    
    
    state_holiday_summary = state_holiday_summary.sort_values(by='Average Sales', ascending=True)
    school_holiday_summary = school_holiday_summary.sort_values(by='Average Sales', ascending=True)
    

    print("Sales Summary by State Holiday (Sorted by Average Sales):")
    display(state_holiday_summary)
    print("\nSales Summary by School Holiday (Sorted by Average Sales):")
    display(school_holiday_summary)
    
   


    logging.info("Holiday impact analysis completed.")
