#!/usr/bin/env python
# coding: utf-8

# # <center>Data Mining Project</center>
# 
# <center>
# Master in Data Science and Advanced Analytics <br>
# NOVA Information Management School
# </center>
# 
# ** **
# ## <center>*ABCDEats Inc*</center>
# 
# <center>
# Group 19 <br>
# Jan-Louis Schneider, 20240506  <br>
# Marta Boavida, 20240519  <br>
# Matilde Miguel, 20240549  <br>
# Sofia Gomes, 20240848  <br>
# </center>
# 
# ** **

# In[40]:


get_ipython().system('jupyter nbconvert --to script "Processing_Data.ipynb"')


# In[41]:


import pandas as pd 
import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import scipy.stats as stats
import warnings

from math import ceil
from sklearn.impute import KNNImputer

from Processing_Data import *

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# In[42]:


df = pd.read_csv("../dataset/df_preprocessing.csv")


# ## <span style="color:salmon">3.1 New Features </span> 

# Creating new features can significantly enhance our analysis by providing additional insights and improving the performance of models

# #### <span style="color:salmon"> 3.1.1 Customer Lifetime  </span>
# Interval of customer activity, so we have an idea of ​​how many days the customer ordered.

# In[43]:


df['lifetime_days'] = df['last_order'] - df['first_order']
df['lifetime_days'].dtype


# #### <span style="color:salmon"> 3.1.2 Most frequent order day of the week  </span>
# Indicates the days of the week on which the customer placed the most orders.

# In[44]:


dows = ['DOW_1', 'DOW_2', 'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6', 'DOW_0'] # this order so it is from Monday to Sunday, not Sunday to Saturday]
def frequent_days(customer):
    max_value = customer[dows].max() # Day with the most orders
    result = []
    for col in dows: # Checks if there is more than one day with max_value
        if customer[col] == max_value:
            result.append(col)
    return result

df['preferred_order_days'] = df.apply(frequent_days, axis=1)
df['preferred_order_days'].dtype # obj 
all(isinstance(i, list) for i in df['preferred_order_days']) # confirm that all values ​​are lists


# #### <span style="color:salmon"> 3.1.3 Most frequent part of the day  </span>
# 6h-12h --> Morning (Breakfast)  
# 12h-18h --> Afternoon (Lunch)  
# 18h-00h --> Evening (Dinner)  
# 00h-6h --> Night

# In[45]:


def part_of_the_day(hour):
    if 6 <= hour < 12:
        return '06h-12h'
    elif 12 <= hour < 18:
        return '12h-18h'
    elif 18 <= hour < 24:
        return '18h-00h'
    else:  # 0 <= hour < 6
        return '00h-06h'

def frequent_hours(customer):
    part_counts = {
        '06h-12h': 0,
        '12h-18h': 0,
        '18h-00h': 0,
        '00h-06h': 0}
    for hour in range(24):
        num_orders = customer[f'HR_{hour}']
        if pd.isna(num_orders): # Ignore NaN
            continue
        part_of_day = part_of_the_day(hour)
        part_counts[part_of_day] += num_orders

    # Part of the day with the highest number of orders
    max_value = 0
    result = []
    for part, count in part_counts.items():
        if count > max_value:
            max_value = count  
            result = [part] 
        elif count == max_value:
            result.append(part) 
    return result
    
df['preferred_part_of_day'] = df.apply(frequent_hours, axis=1)
df['preferred_part_of_day'].dtype # obj 
all(isinstance(i, list) for i in df['preferred_part_of_day']) # confirm that all values ​​are lists


# #### <span style="color:salmon"> 3.1.4 Total monetary units spend </span>
# Sum all total expenses.

# In[46]:


cuisine = df.filter(like='CUI_').columns.tolist() # Types of cuisine
df['total_expenses'] = df[cuisine].sum(axis=1)
df['total_expenses'].dtype


# #### <span style="color:salmon"> 3.1.5 Average monetary units per product </span>
# Show the average monetary of all products.

# In[47]:


df['avg_per_product'] = pd.to_numeric(df['total_expenses'] / df['product_count'].replace(0, pd.NA), errors='coerce')
df['avg_per_product'].dtype


# #### <span style="color:salmon"> 3.1.6 Average monetary units per order </span>
# Show the average monetary per order. 

# In[48]:


df['avg_per_order'] = pd.to_numeric(df['total_expenses'] / df[dows].sum(axis=1).replace(0, pd.NA), errors='coerce')
df['avg_per_order'].dtype


# #### <span style="color:salmon"> 3.1.7 Average order size </span>
# Help identifying users who make larger orders.

# In[49]:


df['avg_order_size'] = pd.to_numeric(df['product_count'] / df[dows].sum(axis=1).replace(0, pd.NA), errors='coerce')
df['avg_order_size'].dtype


# #### <span style="color:salmon"> 3.1.8 Culinary profile </span>
# A proportion of ordered cuisines. A higher number indicates more diversity of types of cuisine you ordered.

# In[50]:


total_cuisine = len(cuisine)

df['culinary_variety'] = round((df[cuisine].gt(0).sum(axis=1) / total_cuisine), 5)
df['culinary_variety'].dtype


# #### <span style="color:salmon"> 3.1.9 Loyalty to chain restaurants </span>
# Proportion of orders from restaurant chains. A high value indicates that you prefer to try different restaurant chains. A lower value is only more faithful to certain chains.

# In[51]:


df['chain_preference'] = pd.to_numeric(df['is_chain'] / df[dows].sum(axis=1).replace(0, pd.NA), errors='coerce')
df['chain_preference'].dtype


# #### <span style="color:salmon"> 3.1.10 Loyalty to venders </span>
# Proportion of orders from specific restaurants. A high value indicates that you prefer to try different restaurants. A lower tend to be more loyal to specific restaurants.

# In[52]:


df['loyalty_to_venders'] = pd.to_numeric(df['vendor_count'] / df[dows].sum(axis=1).replace(0, pd.NA), errors='coerce')
df['loyalty_to_venders'].dtype


# To see all the new features that we added:

# In[53]:


df.head(20)


# ## <span style="color:salmon">3.2 New metric and non metric features </span> 

# New metric and non metric features:

# In[54]:


new_metric_features = ['lifetime_days', 'total_expenses', 'avg_per_product', 'avg_per_order', 'avg_order_size', 'culinary_variety', 'chain_preference', 'loyalty_to_venders']
new_non_metric_features = ['preferred_order_days', 'preferred_part_of_day']
new_features = new_metric_features + new_non_metric_features


# Descriptives:

# In[55]:


df[new_features].describe(include="all").T


# Missing values:

# In[56]:


missing_rows = df[new_features].isna().any(axis=1)


# In[57]:


# Percentage of missing values in each variable:
missing_percentage = ((df[new_features].isnull().sum() / len(df)) * 100).sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

print(f"Percentage of missing values:\n {missing_percentage}")


# Visualize new numerical metric features:

# In[58]:


sns.set()

# Set up the figure and axes
rows, cols = 12, 5 
fig, axes = plt.subplots(rows, cols, figsize=(25, 30))  

# Plot each feature
for ax, feat in zip(axes.flatten(), new_metric_features):
    ax.hist(df[feat], bins=20, color='skyblue', edgecolor='black')  
    ax.set_title(feat, fontsize=10, y=-0.2)  

# Hide unused subplots:
for ax in axes.flatten()[len(new_metric_features):]:
    ax.set_visible(False)

# Set a global title and adjust layout
plt.suptitle("Numeric Variables' Histograms", fontsize=16, y=1.02)  
plt.tight_layout()
plt.show()


# Visualize new non-numerical features:

# In[59]:


for column in new_non_metric_features:
    
    top_categories = df[column].value_counts().head(20)

    top_categories_sorted = top_categories.sort_values(ascending=True)

    data_filtered = df[df[column].isin(top_categories_sorted.index)]
    
   
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data_filtered, 
                  x=column, 
                  order=top_categories_sorted.index,  
                  palette='tab20b')
    
  
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Top 20 Categories in {column}')
    
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Treat missing values in new features:

# In[60]:


# Percentage of missing values in each variable:
missing_percentage = ((df[new_features].isnull().sum() / len(df)) * 100).sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

print(f"Percentage of missing values:\n {missing_percentage}")


# Fill numerical missing values with median:

# In[61]:


median_variables = ['avg_per_product', 'avg_per_order', 'avg_order_size', 'chain_preference', 'loyalty_to_venders']
for column in median_variables:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


# Percentage of missing values in each variable:

# In[62]:


missing_percentage = ((df[new_features].isnull().sum() / len(df)) * 100).sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

print(f"Percentage of missing values:\n {missing_percentage}")


# Store the index of rows with missing values in new_features and filter the DataFrame using the index:

# In[63]:


missing_rows_index = df[missing_rows].index

df_missing = df.loc[missing_rows_index]
df_missing


# Outliers:

# In[64]:


sns.set()


selected_features = new_metric_features

# Set up the figure and axes
rows, cols = 3, 3  #
fig, axes = plt.subplots(rows, cols, figsize=(15, 10)) 

# Flatten axes for iteration
axes = axes.flatten()

# Plot each feature as a box plot
for i, (ax, feat) in enumerate(zip(axes, selected_features)):
    sns.boxplot(data=df, x=feat, ax=ax, color='skyblue')  
    ax.set_title(feat, fontsize=10)  
    
# Hide any unused subplots:
for ax in axes[len(selected_features):]:
    ax.set_visible(False)

# Set a global title and adjust layout
plt.suptitle("Selected Numeric Variables' Box Plots", fontsize=16, y=1.02)  
plt.tight_layout()
plt.show()


# Outlier Removal using automatic method:

# In[65]:


# Compute the interquartile range
q1 = df[new_metric_features].quantile(0.25)
q3 = df[new_metric_features].quantile(0.75)
iqr = q3 - q1

# Compute the limits:
lower_lim = q1 - (1.5 * iqr)
upper_lim = q3 + (1.5 * iqr)

for feature in new_metric_features:
    print(f"{feature:<25}  Lower Limit: {lower_lim[feature].round(5):>10}      Upper Limit: {upper_lim[feature].round(5):>10}")


# Observations in which all features are outliers:

# In[66]:


outliers, obvious_outliers = identify_outliers(df, new_metric_features, lower_lim, upper_lim)


# Conclusion: There is no observation in which all new features is an outlier. 
# 
# There is no outlier in 'lifetime_days', 'chain_preference'.
# 
# Check if there is any observation only with outliers, except on these features.

# In[67]:


new_metric_features_test = ['total_expenses', 'avg_per_product', 'avg_per_order', 'avg_order_size', 'culinary_variety', 'loyalty_to_venders']
outliers, obvious_outliers = identify_outliers(df, new_metric_features_test, lower_lim, upper_lim)


# Conclusion: There is no observation with outliers in all new features.

# Observations in which at least one new feature is an outlier:

# In[68]:


new_filters_iqr = []                                            
for metric in new_metric_features:
    llim = lower_lim[metric]
    ulim = upper_lim[metric]
    new_filters_iqr.append(df[metric].between(llim, ulim, inclusive='neither'))

new_filters_iqr_all = pd.concat(new_filters_iqr, axis=1).all(axis=1)


# In[69]:


new_filters_iqr_all


# In[70]:


# Number of observations with at least one features considered an outlier
new_features_percentage_data_kept = len(df[new_filters_iqr_all])/len(df)*100
new_features_percentage_outliers = round(100 - new_features_percentage_data_kept, 5)
print(f"Percentage of observations with at least one features considered an outlier: {new_features_percentage_outliers}%")
print(f"Percentage of data kept after removing outliers: {new_features_percentage_data_kept}%")


# Outliers removal using manual method:

# In[71]:


filters_manual_new_features = (
                (df["total_expenses"] <= 350) #
                &
                (df["avg_per_product"] <= 22) #??
                &
                (df["avg_per_order"] <= 70)  #50                          
                &
                (df["avg_order_size"] <= 4)                                  
                &
                (df["culinary_variety"] <= 0.7)
                &
                (df["loyalty_to_venders"] >= 0.1)
)                     

df_out_man_new_features = df[filters_manual_new_features]


# In[72]:


# Number of observations with at least one features considered an outlier
new_features_percentage_data_kept_manual = 100*(np.round(df_out_man_new_features.shape[0] / df.shape[0], decimals=5))
new_features_percentage_outliers_manual = round(100 - new_features_percentage_data_kept_manual, 5)
print(f"Percentage of observations with at least one features considered an outlier: {new_features_percentage_outliers_manual}%")
print(f"Percentage of data kept after removing outliers: {new_features_percentage_data_kept_manual}%")


# Remove outliers combining automatic and manual methods:

# In[73]:


df = df[(new_filters_iqr_all | filters_manual_new_features)]


# ## <span style="color:salmon">3.3 Visualize all features </span> 

# In[74]:


all_metric_features = [
    'customer_age', 'vendor_count', 'product_count', 'is_chain', 'first_order', 
    'last_order', 'CUI_American', 'CUI_Asian', 'CUI_Beverages', 'CUI_Cafe', 
    'CUI_Chicken Dishes', 'CUI_Chinese', 'CUI_Desserts', 'CUI_Healthy', 
    'CUI_Indian', 'CUI_Italian', 'CUI_Japanese', 'CUI_Noodle Dishes', 
    'CUI_OTHER', 'CUI_Street Food / Snacks', 'CUI_Thai', 'DOW_0', 'DOW_1', 
    'DOW_2', 'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6', 'HR_0', 'HR_1', 'HR_2', 
    'HR_3', 'HR_4', 'HR_5', 'HR_6', 'HR_7', 'HR_8', 'HR_9', 'HR_10', 'HR_11', 
    'HR_12', 'HR_13', 'HR_14', 'HR_15', 'HR_16', 'HR_17', 'HR_18', 'HR_19', 
    'HR_20', 'HR_21', 'HR_22', 'HR_23', 'lifetime_days', 'total_expenses', 
    'avg_per_product', 'avg_per_order', 'avg_order_size', 'culinary_variety', 
    'chain_preference', 'loyalty_to_venders'
]

all_non_metric_features = [
    'customer_region', 'last_promo', 'payment_method', 
    'preferred_order_days', 'preferred_part_of_day'
]

len(all_metric_features)


# #### <span style="color:salmon"> 3.3.1 Numerical Features </span>

# In[75]:


sns.set()

# Set up the figure and axes
rows, cols = 12, 5  
fig, axes = plt.subplots(rows, cols, figsize=(25, 30)) 

# Plot each feature
for ax, feat in zip(axes.flatten(), all_metric_features):
    ax.hist(df[feat], bins=20, color='skyblue', edgecolor='black')  
    ax.set_title(feat, fontsize=10, y=-0.2)  

# Hide unused subplots:
for ax in axes.flatten()[len(all_metric_features):]:
    ax.set_visible(False)

# Set a global title and adjust layout 
plt.suptitle("Numeric Variables' Histograms", fontsize=16, y=1.02)  
plt.tight_layout() 
plt.show()


# #### <span style="color:salmon"> 3.3.2 Categorical Features </span>

# In[76]:


for column in all_non_metric_features:
    
    top_categories = df[column].value_counts().head(20)

    top_categories_sorted = top_categories.sort_values(ascending=True)

    data_filtered = df[df[column].isin(top_categories_sorted.index)]
    
   
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data_filtered, 
                  x=column, 
                  order=top_categories_sorted.index,  
                  palette='tab20b')
    
  
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Top 20 Categories in {column}')
    
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[77]:


# Store in df_new_features the DataFrame of our dataset df
df_new_features = pd.DataFrame(df)

# Save to CSV
df_new_features.to_csv('../dataset/df_new_features.csv', index=False)

