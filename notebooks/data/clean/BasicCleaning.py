#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# # Dealing with the data

# In[2]:


data = pd.read_csv('bodyfat.csv')
data.shape
data.head()


# In[3]:


#Standardize header names.
def rename_coloumn(data):
    #data = data.rename(columns={'Abdomen': 'Waist'})
    data.rename(columns={'Abdomen': 'Waist'}, inplace=True)
    
def coloumn_names_lwr(data):
    data.columns = map(str.lower, data.columns)
    data.head()
    #data.columns = [x.lower() for x in data.columns]
    
def coloumn_names_captl(data):
    data.columns = map(str.capitalize, data.columns)
    data.head()
    #data.columns = [x.capitalize() for x in data.columns]    
'''
def replace_whitespaces(data):
    cols = []
    for col in data.columns:
        cols.append(col.replace(' ', '_'))
    data.columns = cols
    return data
'''    


# In[4]:


def check_missing_data(data):
    print(f"zero values in the dataset\n: {data.isnull().sum() * 100 / len(data)}")
    print(f"unknown values in the dataset:{data.isna().sum() * 100 / len(data)}")


# In[5]:


#numercial and categorical data

numerical_values = pd.DataFrame()
continuous_values = pd.DataFrame()
categorical_values = pd.DataFrame()

def check_datatypes(data): #returns numerical,continuous and categorical values respectively
    numerical = data.select_dtypes(np.number)
    numerical_continuous = data.select_dtypes(include=['float64'])
    categorical = data.select_dtypes(object)
    return numerical,numerical_continuous,categorical


# Omitting the density coloumn on purpose because to predict the body fat percentage of a person based on the circumferential measurements already available, which is good enough to predict with so much available data.
# 
# I will, create Model only with these data, and predict again with my personal data set for the information randomly to validate the model.

# Fat (%) = [(4*95/density) -4.51 x 100]
# 
# source:https://www.cambridge.org/core/services/aop-cambridge-core/content/view/DA80501B784742B9B2F4F454BDEE923B/S0007114567000728a.pdf/the-assessment-of-the-amount-of-fat-in-the-human-body-from-measurements-of-skinfold-thickness.pdf
# 

# # Exploring the data

# In[6]:


#converting the dataset to metric system
def convert_weight_kg(data):
    data['weight']=data['weight'].apply(lambda x : round((x * 0.453),2))
    
def convert_inch_to_cm(data):
    #drop weight, density and body fat percentage
    # 12 inches --> 30 cm
    #formula X cm = [30/12] * input inches
    df_drops = data[['density','bodyfat','age','weight']]
    df2 = data.drop(['density','bodyfat','age','weight'], axis=1)
    df2 = df2.apply(lambda x : x * 2.5)
    data = pd.concat([df_drops, df2], axis=1, join="outer")
    return data
    


# Finding the Relationships:

# I am concerned about the correlation between the Label = 'Bodyfat%' and the features = [weight,Chest,abdomen, hip, Bicep,Thigh]

# so I am going to find the Correation between them, by dropping the rest as follows

# In[7]:


#def find_correlation(data):
#    df_corr = data.drop(['density','age'], axis=1)
#    sns.heatmap(df_corr.corr())


# I will try to find the highly correlated value to the label bodyfat and try to fit them:
# * In our case we can see that the features, weight,chest,abdomen, hip,thigh are all closely correlated.
# * we will try to find the correlation again by dropping the other fewatures for our consideration

# In[8]:


# add new coloumn waist to hip ratio
def waist_to_hip(data):
    data["waist_to_hip"] = data['waist']/data['hip']


# In[9]:


def finalise_correlation(data, cols_to_drop):
    df_corr = data.drop(cols_to_drop, axis=1)
    sns.heatmap(df_corr.corr(), annot=True)
    return df_corr


# NOTE:we witness here that abdomen circumference somewhat highly correlated and is a key contributor to the Bodyfat Percentage. But according to my Domain knowledge, waist(abdomen) to hip ratio is a significant contributor to calculate the bodyfat percentage.
# so with this in mind I will do some "Feature Engineering", with WHR(waist to hip ratio) as another feature in the table.
# 
# sources:
# 1. https://www.bhf.org.uk/informationsupport/heart-matters-magazine/nutrition/weight/best-way-to-measure-body-fat
#     
# 2. https://www.medicalnewstoday.com/articles/319439#how-does-waist-to-hip-ratio-affect-health
#         

# Conclusion: Since the features Bodyfat and Waist asre highly correlated to the WHR, we will construct a linear regression model around Bodyfat and WHR 

# # Collinearity, Transformation

# Checking the Linear Hypothesis

# In[21]:


# find linear Hypotheis using a scatter plot
def plot_scatter(X,y):
    sns.set()
    sns.scatterplot(X,y)
    plt.tight_layout()
    plt.show()
    
    


# Normalising the Distribution:
# Since we do not have multiple features to predict the label, we don't use any transformation methods liske standard scalar or Min-Max scalar
# 
# we go on to create a 1. model after train test split 2.check the error metrics to check the accuracy 3. save the model 4.use external data on this model to predict the accuracy

# Separate the features from the labels

# In[11]:


#def separate_label_features(data):
#    y = data['TARGET_D']
#    X = data.drop(['TARGET_D'], axis=1)
    
    


# In[12]:


#pd.set_option('display.max_columns', None)


# In[24]:


# Fucntion calls and declarations
numerical_values = pd.DataFrame()
continuous_values = pd.DataFrame()
categorical_values = pd.DataFrame()
cols_to_drop = ['density','age','neck','thigh','height','knee','ankle','forearm','wrist']
df_corr = pd.DataFrame()
data["waist_to_hip"] = ''
rename_coloumn(data)
coloumn_names_lwr(data)#module
check_missing_data(data)#module
numerical_values,continuous_values,categorical_values = check_datatypes(data)#module
convert_weight_kg(data)#module
convert_inch_to_cm(numerical_values)#module
#find_correlation(data)#find_correlation(numerical_values)#input numerical values
waist_to_hip(data)#module
df_corr = finalise_correlation(data, cols_to_drop)#pick highly correlated feature dataframes to predict the label
df_corr


#numerical_values.head(20)


# In[28]:


#plot the correlated features with the label--> plot_scatter(x,y)
plot_scatter(df_corr['waist'],df_corr['bodyfat']) 
plot_scatter(df_corr['waist_to_hip'],df_corr['bodyfat']) 


# plot the features distribution:
# It made no sense to plot tis distribution, mostly because of the limited sample size..

# In[14]:


sns.displot(data=data,x="waist");
plt.show()


# In[15]:


sns.displot(data=data,x="waist_to_hip");
plt.show()


# In[16]:


sns.displot(data=data,x="bodyfat");
plt.show()


# In[ ]:


creating the python modules


# In[ ]:





# Export the clean file

# In[ ]:


result.to_excel('bodyfat_clean_data.csv', header=True)

