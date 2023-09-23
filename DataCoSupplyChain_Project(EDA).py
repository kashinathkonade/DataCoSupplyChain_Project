# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:14:33 2023

@author: kashinath konade
"""

# Business Problem: The Problem associated with data is to predict the sell versus profit or loss
# Business Objective :Maximize the sell 
# Business Contraint:Minimize loss
# Sucess Criteria:
    #1.Business sucess Criteria :Key success factors in supply chain management are the aspects such as quality,cost.
    #2.Economic Success Criteria:Performance imrpovement in terms of customer serviceand satisfaction.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_cosupply = pd.read_csv(r"D:/360DigiTMG Assignment ANS & Key/DataCoSupplyChainDataset Project/DataCoSupplyChainDataset.csv", encoding= 'unicode_escape');
df = pd.DataFrame(data_cosupply)

df.columns # column names

df.shape

df.duplicated().sum()

df.info()  # Names and Data types of Each column

df.isna().sum() #check presence of nan value

df.describe()


df = df.drop(['Product Description','Order Zipcode','Latitude','Longitude',
              'Order City','Order Country','Customer Password','Customer Fname',
              'Customer Lname','Order State','Order Item Quantity',
              'Order Item Cardprod Id','Product Image','Product Card Id',
              'Order Profit Per Order','Order State','shipping date (DateOrders)',
              'Order Customer Id','Department Id','Order Item Product Price',
              'Order Item Total','Customer Email','Customer Street',
              'Late_delivery_risk','Product Status','Customer Zipcode','Product Category Id',
             'Order Item Id','Order Id','Category Id' ], axis = 1)


df.isna().sum()

#As there is no null values , No need of imputing.

# check the outliers with boxplot and remove the outlie

# Boxplot of "Sales per customer","Benefit per order","Order Item Discount",
  #"Order Item Discount Rate","Order Item Profit Ratio",Sales

sns.boxplot(df["Sales per customer"]) #Outliers present, need to remove

sns.boxplot(df["Benefit per order"])  # Outliers present, need to remove


sns.boxplot(df["Order Item Discount"]) # Outliers present

sns.boxplot(df["Order Item Discount Rate"]) # Outliers Absent

sns.boxplot(df["Order Item Profit Ratio"]) # Outliers present

sns.boxplot(df.Sales) # Outliers Present

sns.boxplot(df["Product Price"]) # Outliers Present


# Detection of outliers of Sales per customer (find limits based on IQR)
IQR = df['Sales per customer'].quantile(0.75) - df['Sales per customer'].quantile(0.25)

lower_limit = df['Sales per customer'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Sales per customer'].quantile(0.75) + (IQR * 1.5)

# 1. Remove ( trim the dataset)
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df['Sales per customer'] > upper_limit, True,
                     np.where(df['Sales per customer'] < lower_limit, True, False))

# outliers data
df_out = df.loc[outliers_df, ]

df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed['Sales per customer'])

 


# Detection of outliers of 'Benefit per order' (find limits based on IQR)
IQR = df['Benefit per order'].quantile(0.75) - df['Benefit per order'].quantile(0.25)

lower_limit = df['Benefit per order'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Benefit per order'].quantile(0.75) + (IQR * 1.5)


outliers_df = np.where(df['Benefit per order'] > upper_limit, True,
                     np.where(df['Benefit per order'] < lower_limit, True, False))


# Replace the outliers by the maximum and minimum limit as it still shows some outliers

df['df_replaced'] = pd.DataFrame(np.where(df['Benefit per order'] > upper_limit, upper_limit, np.where(df['Benefit per order'] < lower_limit, lower_limit, df['Benefit per order'])))
sns.boxplot(df.df_replaced)




# Detection of outliers of 'Order Item Discount' (find limits based on IQR)
IQR = df['Order Item Discount'].quantile(0.75) - df['Order Item Discount'].quantile(0.25)

lower_limit = df['Order Item Discount'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Order Item Discount'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit as it still shows some outliers

df['df_replaced'] = pd.DataFrame(np.where(df['Order Item Discount'] > upper_limit, upper_limit, np.where(df['Order Item Discount'] < lower_limit, lower_limit, df['Order Item Discount'])))
sns.boxplot(df.df_replaced)



# Detection of outliers of 'Order Item Profit Ratio'(find limits based on IQR)
IQR = df['Order Item Profit Ratio'].quantile(0.75) - df['Order Item Profit Ratio'].quantile(0.25)

lower_limit = df['Order Item Profit Ratio'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Order Item Profit Ratio'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit as it still shows some outliers

df['df_replaced'] = pd.DataFrame(np.where(df['Order Item Profit Ratio'] > upper_limit, upper_limit, np.where(df['Order Item Profit Ratio'] < lower_limit, lower_limit, df['Order Item Profit Ratio'])))
sns.boxplot(df.df_replaced)




# Detection of outliers of 'Sales' (find limits based on IQR)
IQR = df['Sales'].quantile(0.75) - df['Sales'].quantile(0.25)

lower_limit = df['Sales'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Sales'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit as it still shows some outliers

df['df_replaced'] = pd.DataFrame(np.where(df['Sales'] > upper_limit, upper_limit, np.where(df['Sales'] < lower_limit, lower_limit, df['Sales'])))
sns.boxplot(df.df_replaced)



# Detection of outliers of 'Product Price' (find limits based on IQR)
IQR = df['Product Price'].quantile(0.75) - df['Product Price'].quantile(0.25)

lower_limit = df['Product Price'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Product Price'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit as it still shows some outliers

df['df_replaced'] = pd.DataFrame(np.where(df['Product Price'] > upper_limit, upper_limit, np.where(df['Product Price'] < lower_limit, lower_limit, df['Product Price'])))
sns.boxplot(df.df_replaced)

df


# Date orders column

df['order date (DateOrders)'].dtype

df['order date (DateOrders)'].sample(10)

df[['date1', 'date2', 'date3']] = df['order date (DateOrders)'].str.split('/', expand = True)
df[['date4', 'date5']] = df['date3'].str.split(':', expand = True)
df[['date6', 'date7']] = df['date4'].str.split(' ', expand = True)
df.sample(30)

df.drop(['date2', 'date3', 'date4', 'date5', 'date7', 'order date (DateOrders)'], axis = 1, inplace = True)
df = df.rename(columns = {'date6' : 'Order Year', 'date1': 'Order Month'})
df[['Order Year', 'Order Month']].dtypes

df.drop(['Order Month'], axis = 1, inplace = True)

# Review columns:

df.info()

# Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking whether data is normally distributed

stats.probplot(df.Sales, dist = "norm", plot = pylab)

stats.probplot(df['Product Price'], dist = "norm", plot = pylab)

stats.probplot(df['Sales per customer'], dist = "norm", plot = pylab)

stats.probplot(df['Benefit per order'], dist = "norm", plot = pylab)

stats.probplot(df['Order Item Profit Ratio'], dist = "norm", plot = pylab)

stats.probplot(df['Order Item Discount'], dist = "norm", plot = pylab)

stats.probplot(df['Order Item Discount Rate'], dist = "norm", plot = pylab)



from sklearn.preprocessing import StandardScaler

# Initialise the Scaler
scaler = StandardScaler()


df[["Benefit per order","Sales per customer","Order Item Discount","Order Item Discount Rate",
    "Order Item Profit Ratio","Sales","Product Price"]] = scaler.fit_transform(df[["Benefit per order","Sales per customer","Order Item Discount","Order Item Discount Rate",
        "Order Item Profit Ratio","Sales","Product Price"]])
dataset = pd.DataFrame(df)

res = dataset.describe()





# Label Encoder
from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

df.columns


# Data Split into Input and Output variables
X = dataset.iloc[:,:29]
y = dataset.iloc[:,10]
Z = dataset.iloc[:,12]

# Converting Categorical data into Numerical data by Label Encoder

X['Type'] = labelencoder.fit_transform(X['Type'])
X['Delivery Status'] = labelencoder.fit_transform(X['Delivery Status'])
X['Category Name'] = labelencoder.fit_transform(X['Category Name'])
X['Customer City'] = labelencoder.fit_transform(X['Customer City'])
X['Customer Country'] = labelencoder.fit_transform(X['Customer Country'])
X['Customer Segment'] = labelencoder.fit_transform(X['Customer Segment'])
X['Department Name'] = labelencoder.fit_transform(X['Department Name'])

X['Order Region'] = labelencoder.fit_transform(X['Order Region'])
X['Market'] = labelencoder.fit_transform(X['Market'])
X['Order Status'] = labelencoder.fit_transform(X['Order Status'])
X['Shipping Mode'] = labelencoder.fit_transform(X['Shipping Mode'])

# final output of data cosupply chain
X



#Conclusion 

#1] Days for shipping and scheduled days are quite different ,need more accurate to pradict
#2] Motly people place one item in an order which is understood
#3] And The Profit has negative values around mean value which means from same orders,the profit is very less than mean values
#4] Profit per order has peak around zero,not good for business.
#5] Sales are more in same of the customer segment but we should work on other segment to increase the sales.
