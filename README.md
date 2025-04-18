import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# STEP 1: Load the CSV dataset
# =====================================
# Ensure this file exists: data/Assignment_Data for 1a.csv
df = pd.read_csv('data/Assignment_Data for 1a.csv', parse_dates=['booking_time'])

# =====================================
# STEP 2: Driver & Food Delivery Analysis
# =====================================

# Filter transactions on 20 January 2021
df_20jan = df[df['booking_time'].dt.date == pd.to_datetime('2021-01-20').date()]

# Display available service types for that date
print("Available service types on 2021-01-20:", df_20jan['service_type'].unique())

# Identify food delivery drivers (e.g., GO-FOOD)
food_service = ['GO-FOOD']  # Change this if your food delivery label is different
food_drivers = df_20jan[df_20jan['service_type'].isin(food_service)]['driver_id'].unique()

# Identify all active drivers on that date
all_drivers = df_20jan['driver_id'].unique()

# Calculate proportion of drivers who delivered food
food_driver_proportion = len(food_drivers) / len(all_drivers)

print(f"Proportion of drivers delivering only food on January 20, 2021: {food_driver_proportion:.2%}")

# =====================================
# STEP 3: RFM Scoring
# =====================================

# Calculate Recency, Frequency, and Monetary values per customer
current_date = df['booking_time'].max()

rfm = df.groupby('customer_id').agg({
    'booking_time': lambda x: (current_date - x.max()).days,  # Recency
    'order_no': 'count',                                       # Frequency
    'actual_gmv': 'sum'                                        # Monetary
}).reset_index()

rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# Replace zero monetary values to avoid issues
rfm['Monetary'] = rfm['Monetary'].replace(0, 1)

# Assign scores using equal-width binning (1–4)
rfm['R_Score'] = pd.cut(rfm['Recency'], bins=4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=4, labels=[1, 2, 3, 4])

# Combine RFM scores into a single string
rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

# Define customer class based on monetary quartiles
monetary_bins = [
    -1,
    rfm['Monetary'].quantile(0.25) + 1e-6,
    rfm['Monetary'].quantile(0.5) + 2e-6,
    rfm['Monetary'].quantile(0.75) + 3e-6,
    rfm['Monetary'].max() + 4e-6
]

monetary_labels = ['Low Spender', 'Medium Spender', 'High Spender', 'Top Spender']

rfm['Customer_Class'] = pd.cut(
    rfm['Monetary'],
    bins=monetary_bins,
    labels=monetary_labels,
    include_lowest=True
)

# =====================================
# STEP 4: Prepare Data for Visualization
# =====================================

# Recency: daily active customers
date_recency = df.groupby(df['booking_time'].dt.date)['customer_id'].nunique().reset_index()
date_recency.columns = ['Date', 'Active_Customers']

# Frequency: total orders per service type
service_freq = df.groupby('service_type')['order_no'].count().reset_index()
service_freq.columns = ['Service_Type', 'Order_Count']

# Monetary: top customer classifications
customer_spending = rfm[['customer_id', 'Customer_Class']]

# =====================================
# STEP 5: Visualize RFM Components
# =====================================

sns.set(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency
sns.lineplot(data=date_recency, x='Date', y='Active_Customers', ax=axes[0])
axes[0].set_title('Daily Active Customers (Recency)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Active Customers')
axes[0].tick_params(axis='x', rotation=45)

# Frequency
sns.barplot(data=service_freq, x='Service_Type', y='Order_Count', ax=axes[1])
axes[1].set_title('Order Frequency by Service Type')
axes[1].set_xlabel('Service Type')
axes[1].set_ylabel('Order Count')
axes[1].tick_params(axis='x', rotation=45)

# Monetary
top_customers = customer_spending.head(20)
sns.barplot(data=top_customers, x='customer_id', y='Customer_Class', ax=axes[2])
axes[2].set_title('Top 20 Customers by Spending Class')
axes[2].set_xlabel('Customer ID')
axes[2].set_ylabel('Customer Class')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# =====================================
# STEP 6: Display Final Output
# =====================================
print("Customer RFM Summary with Classification:")
print(rfm.head())
