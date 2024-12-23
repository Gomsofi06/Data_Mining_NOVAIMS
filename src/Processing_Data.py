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
# 

# ## <span style="color:salmon"> Notebook </span> 
# 
# In this notebook, after exploring our data and removing the duplicates in the other notebook, we are going to search for incoherencies, handle the missing values that can be imputed with simple techniques that do not tend to leak data, remove outliers and explore the results of our preprocessing.

# ## <span style="color:salmon"> Table of Contents </span>
# 
# <a class="anchor" id="top"></a>
# 
# 1. [Import Libraries](#one-bullet) <br>
# 
# 2. [Import Datasets](#two-bullet) <br>
# 
# 3. [Data Types](#three-bullet) <br>
# 
# 4. [Incoherencies](#four-bullet) <br>
# 
# 5. [Handling Missing Values](#five-bullet) <br>
# 
# 6. [Removing Outliers  & Changing Distributions](#six-bullet) <br>
# 
# 7. [Export Datasets](#seven-bullet) <br> 
# 

# <a class="anchor" id="one-bullet"></a>
# ## <span style="color:salmon"> 1. Import Libraries </span> 

# In[1]:


import pandas as pd 
import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import scipy.stats as stats
import warnings

from utils import *
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from math import ceil
from sklearn.impute import KNNImputer

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# <a class="anchor" id="two-bullet"> 
# 
# ## <span style="color:salmon"> 2. Import Dataset </span> 
# 
# <a href="#top">Top &#129033;</a>

# In[2]:


df = pd.read_csv("../dataset/df_explore.csv")


# <a class="anchor" id="three-bullet"> 
# 
# ## <span style="color:salmon">3. Data types </span> 
# 
# <a href="#top">Top &#129033;</a>

# In[3]:


numerical_features = ['customer_age', 'vendor_count', 'product_count', 'is_chain',
                      'first_order', 'last_order', 'CUI_American', 'CUI_Asian',
                      'CUI_Beverages', 'CUI_Cafe', 'CUI_Chicken Dishes', 'CUI_Chinese',
                      'CUI_Desserts', 'CUI_Healthy', 'CUI_Indian', 'CUI_Italian',
                      'CUI_Japanese', 'CUI_Noodle Dishes', 'CUI_OTHER',
                      'CUI_Street Food / Snacks', 'CUI_Thai', 'DOW_0', 'DOW_1', 'DOW_2',
                      'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6', 'HR_0', 'HR_1', 'HR_2', 'HR_3',
                      'HR_4', 'HR_5', 'HR_6', 'HR_7', 'HR_8', 'HR_9', 'HR_10', 'HR_11',
                      'HR_12', 'HR_13', 'HR_14', 'HR_15', 'HR_16', 'HR_17', 'HR_18', 'HR_19',
                      'HR_20', 'HR_21', 'HR_22', 'HR_23'
                     ]


# In[4]:


numerical_df = df[numerical_features]


# As all necessary analysis and explanations were given in the previous notebook, this notebook will only have the necessary changes on the data types.

# In[5]:


float_to_int(df, ['customer_age', 'first_order', 'HR_0'])


# <a class="anchor" id="four-bullet"> 
# 
# ## <span style="color:salmon"> 4. Incoherencies</span> 
# 
# <a href="#top">Top &#129033;</a>

# ## <span style="color:yellow"> O QUE FAZER COM AS INCOERENCIAS?</span> 
# 
# ## <span style="color:yellow"> CHECK FOR MORE INCOERENCIAS</span> 

# 1. The mean of product_count is 5 and exists one number with value 269

# In[6]:


df.loc[(df["product_count"]==269)]


# 2. the column HR_0 is the only with 0 requests

# In[7]:


df['HR_0'].max()


# 3. Product_count == 0 and vendor_count >= 1:

# In[8]:


df.loc[(df["product_count"]==0) & (df["vendor_count"]>=1)]


# Replace product_count = 0 with NaN for these rows:

# In[9]:


df.loc[(df["product_count"] == 0) & (df["vendor_count"] >= 1), "product_count"] = np.nan


# In[10]:


product_vendor_count_missing = df.loc[(df["product_count"].isna()) & (df["vendor_count"] >= 1)]
product_vendor_count_missing


# <a class="anchor" id="five-bullet"> 
# 
# ## <span style="color:salmon"> 5. Handling Missing Values</span> 
# 
# <a href="#top">Top &#129033;</a>

# Dealing with missing values effectively is crucial to ensure our dataset's integrity and the accuracy of your analysis. 
# 
# So, first we check the percentage of missing values:

# In[11]:


missing_percentage = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)

missing_percentage[missing_percentage > 0].sort_values(ascending=False)

print(f"Percentage of missing values:\n{missing_percentage}")


# 1. In case of numerical features, the strategie we used to deal with missing values is input with the median:

# In[12]:


# Input missing values in numerical features using median:
median_variables = ["customer_age", "HR_0"]
for column in median_variables:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


# In[13]:


missing_percentage = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

print(f"Percentage of missing values:\n {missing_percentage}")


# In order to threat the missing values in first_order, we use the technique KNNImputer, which for each point with missing values, finds the K nearest neighbors based on a distance metric and replaces the missing value with the average value of the K nearest neighbors. 

# In[14]:


# Input missing values in first_order using knn:

knn_rows = df.loc[df["first_order"].isna()]

features_for_imputation = ["last_order", "product_count", "vendor_count", "DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "DOW_5", "DOW_6",  
                            "HR_0", "HR_1", "HR_2", "HR_3", "HR_4", "HR_5", "HR_6", "HR_7", "HR_8", "HR_9", "HR_10", "HR_11", "HR_12", "HR_13", 
                            "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23",
                            "CUI_American", "CUI_Asian", "CUI_Beverages", "CUI_Cafe", "CUI_Chicken Dishes", "CUI_Chinese", "CUI_Desserts", 
                            "CUI_Healthy", "CUI_Indian", "CUI_Italian", "CUI_Japanese", "CUI_Noodle Dishes", "CUI_OTHER", "CUI_Street Food / Snacks", "CUI_Thai"]

imputation_data = df.loc[df["first_order"].isna(), features_for_imputation]

knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")

imputed_values = knn_imputer.fit_transform(imputation_data)

df.loc[df["first_order"].isna(), "first_order"] = imputed_values[:, 0]  

print("Missing values in first_order have been imputed using KNN.")


# Input missing values in product_count using KNNImputer:

# In[15]:


# Select rows where "product_count" is NaN
knn_rows = df.loc[df["product_count"].isna()]

# Drop the unnecessary columns from numerical_df
features_for_imputation = numerical_df.drop(
    columns=[
        "product_count", "customer_age", "is_chain", "CUI_American", "CUI_Asian", 
        "CUI_Beverages", "CUI_Cafe", "CUI_Chicken Dishes", "CUI_Chinese", "CUI_Desserts", 
        "CUI_Healthy", "CUI_Indian", "CUI_Italian", "CUI_Japanese", "CUI_Noodle Dishes", 
        "CUI_OTHER", "CUI_Street Food / Snacks", "CUI_Thai"
    ]
)

# Extract the rows for imputation
imputation_data = df.loc[df["product_count"].isna(), features_for_imputation.columns]

# Perform KNN imputation
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputed_values = knn_imputer.fit_transform(imputation_data)

# Assign the imputed values back to the DataFrame
df.loc[df["product_count"].isna(), "product_count"] = imputed_values[:, 0]

print("Missing values in product_count have been imputed using KNN.")


# 2. In case of categorical features, the strategie we used to deal with missing values is input with the mode:

# In[16]:


# Input missing values in categorical features using mode:
mode_variables = ["last_promo", "customer_region"]
for column in mode_variables:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)


# To check if every missing value, has been taken.

# In[17]:


missing_percentage = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

print(f"Percentage of missing values:\n {missing_percentage}")


# <a class="anchor" id="six-bullet"> 
# 
# ## <span style="color:salmon"> 6. Removing Outliers</span> 
# 
# <a href="#top">Top &#129033;</a>

# Outliers are data points that deviate significantly from the rest of the observations in a dataset.
# 
#  They can result from variability in the data or errors during data collection, entry, or processing.
# 
# So, we have to threat them.

# 1. Outliers visualization

# In[18]:


sns.set()

# Set up the figure and axes
rows, cols = 13, 4  
fig, axes = plt.subplots(rows, cols, figsize=(25, 30))  

# Plot each feature as a box plot
for ax, feat in zip(axes.flatten(), numerical_df):
    sns.boxplot(data=df, x=feat, ax=ax, color='skyblue')  
    ax.set_title(feat, fontsize=10, y=-0.2)  #

# Hide unused subplots:
for ax in axes.flatten()[len(numerical_df):]:
    ax.set_visible(False)

# Set a global title and adjust layout
plt.suptitle("Numeric Variables' Box Plots", fontsize=16, y=1.02)  
plt.tight_layout()
plt.show()


# ## <span style="color:yellow"> TRATAR OUTLIERS INDIVIDUALMENTE</span> 

# 2. Outliers removal

# There is two methods to treat with outliers:
# - Automatic method 
# - Manual method
# 
# We will try those two and compare which is better in our database.

# - Using the Automatic method:

# In[19]:


# Compute the inter-quartile range
q1 = df[numerical_features].quantile(0.25)
q3 = df[numerical_features].quantile(0.75)
iqr = q3 - q1

# Compute the limits:
lower_lim = q1 - (1.5 * iqr)
upper_lim = q3 + (1.5 * iqr)

for feature in numerical_features:
    print(f"{feature:<25}  Lower Limit: {lower_lim[feature]:>10}      Upper Limit: {upper_lim[feature]:>10}")


# Observations in which all features are outliers:

# In[20]:


def identify_outliers(dataframe, metric_features, lower_lim, upper_lim):
    outliers = {}
    obvious_outliers = []

    for metric in metric_features:
        if metric not in dataframe.columns:
            continue
        
        if metric not in lower_lim or metric not in upper_lim:
            continue
        
        outliers[metric] = []
        llim = lower_lim[metric]
        ulim = upper_lim[metric]
        
        for i, value in enumerate(dataframe[metric]):
            if pd.isna(value):
                continue
            
            if value < llim or value > ulim:
                outliers[metric].append(value)
        
        print(f"Total outliers in {metric}: {len(outliers[metric])}")

    # Check for observations that are outliers in all features (Obvious Outliers)
    for index, row in dataframe.iterrows():
        is_global_outlier = True
        for metric in metric_features:
            if metric not in dataframe.columns or metric not in lower_lim or metric not in upper_lim:
                is_global_outlier = False
                break
            
            value = row[metric]
            if pd.isna(value):
                is_global_outlier = False
                break
            
            llim = lower_lim[metric]
            ulim = upper_lim[metric]
            
            if llim <= value <= ulim:
                is_global_outlier = False
                break
        
        if is_global_outlier:
            obvious_outliers.append(index)
    print("-----------------------------")
    print(f"Total global outliers: {len(obvious_outliers)}")
    return outliers, obvious_outliers
    
    
outliers, obvious_outliers = identify_outliers(df, numerical_features, lower_lim, upper_lim)


# Conclusion: There is no observation in which all features are outliers. Since there is no outlier in 'HR_0', 'last_order', 'first_order'.
# 
# Check if there is any observation only with outliers, except on these features.

# In[21]:


metric_features_test = numerical_df.drop(columns=['HR_0', 'last_order', 'first_order'])

outliers, obvious_outliers = identify_outliers(df, metric_features_test, lower_lim, upper_lim)


# Conclusion: There is no observation with outliers in all features.

# Observations in which at least one feature is an outlier:

# In[22]:


filters_iqr = []                                            
for metric in numerical_features:
    llim = lower_lim[metric]
    ulim = upper_lim[metric]
    filters_iqr.append(df[metric].between(llim, ulim, inclusive='neither'))

filters_iqr_all = pd.concat(filters_iqr, axis=1).all(axis=1)


# In[23]:


filters_iqr_all


# In[24]:


len(df[~filters_iqr_all])
# Number of observations with at least one features considered an outlier
percentage_outliers = len(df[filters_iqr_all])/len(df)*100
percentage_data_kept = round(100 - percentage_outliers, 5)
print(f"Percentage of observations with at least one features considered an outlier: {percentage_outliers}%")
print(f"Percentage of data kept after removing outliers: {percentage_data_kept}%")


# Conclusion: All observations have outliers in some feature

# - Using the manual method:

# In[25]:


filters_manual1 = (
                (df["customer_age"] <= 70) #??
                &
                (df["vendor_count"] <= 30) #??
                &
                (df["product_count"] <= 100)                            
                &
                (df["is_chain"] <= 55)                                  
                &
                (df["CUI_American"] <= 150)
                &
                (df["CUI_Asian"] <= 450)                            
                &
                (df["CUI_Beverages"] <= 150)  
                &
                (df["CUI_Cafe"] <= 140)
                &
                (df["CUI_Chicken Dishes"] <= 70)                            
                &
                (df["CUI_Chinese"] <= 200)                                  
                &
                (df["CUI_Desserts"] <= 80)
                &
                (df["CUI_Healthy"] <= 150)                            
                &
                (df["CUI_Indian"] <= 150)
                &
                (df["CUI_Italian"] <= 200)
                &
                (df["CUI_Japanese"] <= 150)
                &
                (df["CUI_Noodle Dishes"] <= 90)
                &
                (df["CUI_OTHER"] <= 180)
                &
                (df["CUI_Street Food / Snacks"] <= 210)
                &
                (df["CUI_Thai"] <= 70)
                &
                (df["DOW_0"] <= 12)
                &
                (df["DOW_1"] <= 14)
                &
                (df["DOW_2"] <= 10) #(??)
                &
                (df["DOW_3"] <= 12)
                &
                (df["DOW_4"] <=12) #(??)
                &
                (df["DOW_5"] <= 12) #(??)
                &
                (df["DOW_6"] <= 15)
                &
                (df["HR_1"] <= 10)                                  
                &
                (df["HR_2"] < 8)
                &
                (df["HR_3"] <= 8) #(??)                            
                &
                (df["HR_4"] <= 8)                       
                &
                (df["HR_5"] <= 5)                                  
                &
                (df["HR_6"] <= 6)
                &
                (df["HR_7"] <= 10)                            
                &
                (df["HR_8"] <= 19)  
                &
                (df["HR_9"] <= 12)
                &
                (df["HR_10"] < 15)                            
                &
                (df["HR_11"] <= 15)                                  
                &
                (df["HR_12"] <= 15)
                &
                (df["HR_13"] <= 10)                            
                &
                (df["HR_14"] <= 10) 
                &
                (df["HR_15"] <= 12)  
                &
                (df["HR_16"] <= 15)
                &
                (df["HR_17"] <= 16)
                &
                (df["HR_18"] <= 15)                                  
                &
                (df["HR_19"] <= 15)
                &
                (df["HR_20"] <= 15)                            
                &
                (df["HR_21"] <= 7)
                &
                (df["HR_22"] <= 8)
                &
                (df["HR_23"] <= 7)    
)                     

df_out_man1 = df[filters_manual1]


# In[26]:


print('Percentage of data kept after removing outliers:', 100*(np.round(df_out_man1.shape[0] / df.shape[0], decimals=5)))


# In[27]:


filters_manual2 = (
                (df["customer_age"] <= 70) 
                &
                (df["vendor_count"] <= 35) 
                &
                (df["product_count"] <= 80)                          
                &
                (df["is_chain"] <= 40)                                 
                &
                (df["CUI_American"] <= 100)
                &
                (df["CUI_Asian"] <= 300) #200                           
                &
                (df["CUI_Beverages"] <= 100)  
                &
                (df["CUI_Cafe"] <= 120)  
                &
                (df["CUI_Chicken Dishes"] <= 55) #50                           
                &
                (df["CUI_Chinese"] <= 150)                                
                &
                (df["CUI_Desserts"] <= 70) #100
                &
                (df["CUI_Healthy"] <= 120)                           
                &
                (df["CUI_Indian"] <= 120)  
                &
                (df["CUI_Italian"] <= 150)
                &
                (df["CUI_Japanese"] <= 120) 
                &
                (df["CUI_Noodle Dishes"] <= 70)
                &
                (df["CUI_OTHER"] <= 120)
                &
                (df["CUI_Street Food / Snacks"] <= 200)
                &
                (df["CUI_Thai"] <= 60)
                &
                (df["DOW_0"] <= 12) #8
                &
                (df["DOW_1"] <= 12) #8
                &
                (df["DOW_2"] <= 12) #8
                &
                (df["DOW_3"] <= 12) #8
                &
                (df["DOW_4"] <=12) #8
                &
                (df["DOW_5"] <= 13) #8
                &
                (df["DOW_6"] <= 13) #10
                &
                (df["HR_1"] <= 8)                                 
                &
                (df["HR_2"] < 8) 
                &
                (df["HR_3"] <= 8)                             
                &
                (df["HR_4"] < 10)                       
                &
                (df["HR_5"] < 5)                                  
                &
                (df["HR_6"] <= 5)
                &
                (df["HR_7"] <= 5)                            
                &
                (df["HR_8"] <= 15)  #10
                &
                (df["HR_9"] <= 13) #10
                &
                (df["HR_10"] < 15) #10                         
                &
                (df["HR_11"] <= 15)  #10                                
                &
                (df["HR_12"] < 10) #10
                &
                (df["HR_13"] < 8)  #6                          
                &
                (df["HR_14"] < 8) 
                &
                (df["HR_15"] < 10)  
                &
                (df["HR_16"] <= 15) #10
                &
                (df["HR_17"] < 15) #10
                &
                (df["HR_18"] < 12) #10                                 
                &
                (df["HR_19"] < 15) 
                &
                (df["HR_20"] < 10)                            
                &
                (df["HR_21"] < 6)
                &
                (df["HR_22"] < 8)
                &
                (df["HR_23"] < 6)    
)                     

df_out_man2 = df[filters_manual2]


# In[28]:


print('Percentage of data kept after removing outliers:', 100*(np.round(df_out_man2.shape[0] / df.shape[0], decimals=5)))


# In[29]:


filters_manual3 = (
                (df["customer_age"] <= 70) 
                &
                (df["vendor_count"] <= 35) 
                &
                (df["product_count"] <= 80)                          
                &
                (df["is_chain"] <= 40)                                 
                &
                (df["CUI_American"] <= 100)
                &
                (df["CUI_Asian"] <= 200) #200                           
                &
                (df["CUI_Beverages"] <= 100)  
                &
                (df["CUI_Cafe"] <= 120)  
                &
                (df["CUI_Chicken Dishes"] <= 50) #50                           
                &
                (df["CUI_Chinese"] <= 150)                                
                &
                (df["CUI_Desserts"] <= 100) #100
                &
                (df["CUI_Healthy"] <= 120)                           
                &
                (df["CUI_Indian"] <= 120)  
                &
                (df["CUI_Italian"] <= 150)
                &
                (df["CUI_Japanese"] <= 120) 
                &
                (df["CUI_Noodle Dishes"] <= 70)
                &
                (df["CUI_OTHER"] <= 120)
                &
                (df["CUI_Street Food / Snacks"] <= 200)
                &
                (df["CUI_Thai"] <= 60)
                &
                (df["DOW_0"] <= 8) #8
                &
                (df["DOW_1"] <= 8) #8
                &
                (df["DOW_2"] <= 8) #8
                &
                (df["DOW_3"] <= 8) #8
                &
                (df["DOW_4"] <=8) #8
                &
                (df["DOW_5"] <=8) #8
                &
                (df["DOW_6"] <= 10) #10
                &
                (df["HR_1"] <= 8)                                 
                &
                (df["HR_2"] < 8) 
                &
                (df["HR_3"] <= 8)                             
                &
                (df["HR_4"] < 10)                       
                &
                (df["HR_5"] < 5)                                  
                &
                (df["HR_6"] <= 5)
                &
                (df["HR_7"] <= 5)                            
                &
                (df["HR_8"] <= 10)  #10
                &
                (df["HR_9"] <= 10) #10
                &
                (df["HR_10"] < 10) #10                         
                &
                (df["HR_11"] <= 10)  #10                                
                &
                (df["HR_12"] < 10) #10
                &
                (df["HR_13"] < 6)  #6                          
                &
                (df["HR_14"] < 8) 
                &
                (df["HR_15"] < 10)  
                &
                (df["HR_16"] <= 10) #10
                &
                (df["HR_17"] < 10) #10
                &
                (df["HR_18"] < 10) #10                                 
                &
                (df["HR_19"] < 15) 
                &
                (df["HR_20"] < 10)                            
                &
                (df["HR_21"] < 6)
                &
                (df["HR_22"] < 8)
                &
                (df["HR_23"] < 6)    
)                     

df_out_man3 = df[filters_manual3]


# In[30]:


# Number of observations with at least one features considered an outlier
percentage_data_kept_manual = 100*(np.round(df_out_man3.shape[0] / df.shape[0], decimals=5))
percentage_outliers_manual = round(100 - percentage_data_kept_manual, 5)
print(f"Percentage of observations with at least one features considered an outlier: {percentage_outliers_manual}%")
print(f"Percentage of data kept after removing outliers: {percentage_data_kept_manual}%")


# We conclude that we should see what is common to both methods and remove only that because the automatic method is removing a very high percentage of the data.

# In[31]:


df = df[(filters_iqr_all | filters_manual3)]


# In[32]:


sns.set()

# Set up the figure and axes
rows, cols = 13, 4  
fig, axes = plt.subplots(rows, cols, figsize=(25, 30))  

# Plot each feature as a box plot
for ax, feat in zip(axes.flatten(), numerical_df):
    sns.boxplot(data=df, x=feat, ax=ax, color='skyblue')  
    ax.set_title(feat, fontsize=10, y=-0.2)  #

# Hide unused subplots:
for ax in axes.flatten()[len(numerical_df):]:
    ax.set_visible(False)

# Set a global title and adjust layout
plt.suptitle("Numeric Variables' Box Plots", fontsize=16, y=1.02)  
plt.tight_layout()
plt.show()


# <a class="anchor" id="seven-bullet"> 
# 
# ## <span style="color:salmon"> 7. Export Datasets</span> 
# 
# <a href="#top">Top &#129033;</a>

# In[33]:


# Store in df_preprocessing the DataFrame of our dataset df
df_preprocessing = pd.DataFrame(df)

# Save to CSV
df_preprocessing.to_csv('../dataset/df_preprocessing.csv', index=False)

