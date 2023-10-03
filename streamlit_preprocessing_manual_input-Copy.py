#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import pdist, minkowski
from prophet import Prophet
from sklearn.impute import KNNImputer
from collections import defaultdict
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
#if we want to display all columns and rows
pd.pandas.set_option('display.max_columns', None)
#pd.pandas.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.simplefilter('ignore')
import streamlit as st
import io


# In[ ]:


#Describing the data


def describe_data(df):
    st.write("View the cross section of data:")
    st.write(df.head(10))
    st.write("Shape of the dataframe:", df.shape)
    st.write("Datatypes of all features:")
    st.write(df.dtypes)
    st.write("Description of the dataset:")
    st.write(df.describe(include='all').transpose())


# In[ ]:


#Bining the columns into 4 different categories (unique, boolean, categorical,continuous) based on the number of instances in each category 


def check_column_category(df):
    unique=[]
    bool_col=[]
    cat_col=[]
    cont_col =[]
    date_col=[]
    for i in df.columns:
        if df[i].dtype not in ['int64', 'object', 'float64']:
            date_col.append(i)
        else:
            n_values = df[i].nunique()
            if n_values==1:
                unique.append(i)
            elif n_values == 2:
                bool_col.append(i)
            elif n_values < 10:
                cat_col.append(i)
            else:
                cont_col.append(i)

    st.subheader("Check Column Category")
#     unique, bool_col, cat_col, cont_col = check_column_category(df)
    st.write("Unique columns in the data are :", unique)
    st.write("Boolean columns in the data are :", bool_col)
    st.write("Catagorical columns in the data are :", cat_col)
    st.write("Continuious columns in the data are :", cont_col)
    st.write("Datetime columns in the data are:", date_col)


# In[ ]:

def change_dtype(df):
    st.write("Columns in dataframe:\n")
    lst=[]
    for i, col in enumerate(df.columns):
        st.write(i+1,". ",col, " ", df[col].dtype)
        lst.append((i+1,col))
    str1=st.text_input("Please Enter the list of column number's for changing datatype","E.g. 1 3")
    order=str1.split(' ')
    for i in order:
        new_type=st.text_input("Please input the new datatype for column {}".format(lst[int(i)-1][1]), "E.g. int64")
        df[lst[int(i)-1][1]] = df[lst[int(i)-1][1]].astype(new_type)
    return df

def change_dtype1(df, df1):
    st.write("Columns in dataframe:\n")
    lst=[]
    for i, col in enumerate(df.columns):
        st.write(i+1,". ",col, " ", df[col].dtype)
        lst.append((i+1,col))
    str1=st.text_input("Please Enter the list of column number's for changing datatype","E.g. 1 3")
    order=str1.split(' ')
    for i in order:
        new_type=st.text_input("Please input the new datatype for column {}".format(lst[int(i)-1][1]), "E.g. int64")
        df[lst[int(i)-1][1]] = df[lst[int(i)-1][1]].astype(new_type)
        df1[lst[int(i)-1][1]] = df1[lst[int(i)-1][1]].astype(new_type)
    return df, df1

def view_data(df):
    st.header("Choose your filters")
#         # Choose Rows
#     start_row = st.number_input("Enter the starting row number:", min_value=0, value=0)
#     end_row = st.number_input("Enter the ending row number:", min_value=start_row, value=df.shape[0])

    # Choose Columns
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to view:", all_columns, default=all_columns)
    
    selected_columns1=df.select_dtypes(include='object').columns.tolist()
    selected_categories = {}
    for col in selected_columns1:
        selected_categories[col] = st.multiselect(f"Select categories for {col}", df[col].unique().tolist(), default=df[col].unique().tolist())

        
    filtered_df = df.copy()
    filtered_df = filtered_df[selected_columns]
    for col, value in selected_categories.items():
        filtered_df = filtered_df[filtered_df[col].isin(value)]
    sort_by = st.selectbox("Sort by:", ["No sorting", *selected_columns])
    if sort_by != "No sorting":
        ascending = st.selectbox("Order:", ["Ascending", "Descending"])
        ascending = True if ascending == "Ascending" else False
        filtered_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
        
     # Choose Filters
    filters = []
    for col in  filtered_df.select_dtypes(include=['int', 'float']).columns:
        filter_by = st.selectbox(f"Filter {col} by:", ["No Filter", "Above Threshold", "Below Threshold", "Between Thresholds"])
        if filter_by == "No Filter":
            pass
        elif filter_by == "Above Threshold":
            thresh = st.number_input(f"Enter threshold for {col}:")
            filters.append(f"{col} > {thresh}")
        elif filter_by == "Below Threshold":
            thresh = st.number_input(f"Enter threshold for {col}:")
            filters.append(f"{col} < {thresh}")
        else:
            low_thresh = st.number_input(f"Enter lower threshold for {col}:")
            high_thresh = st.number_input(f"Enter upper threshold for {col}:")
            filters.append(f"{col} >= {low_thresh} & {col} <= {high_thresh}")

    # Apply Filters and Sort by
    if filters:
        filtered_df = filtered_df.query(" & ".join(filters))
    start_row = st.number_input("Enter the starting row number:", min_value=0, value=0)
    end_row = st.number_input("Enter the ending row number:", min_value=start_row, value=filtered_df.shape[0])
    filtered_df = filtered_df[start_row:end_row]

    # Show Resultant Dataframe
    st.write("Resultant Dataframe:")
    st.dataframe(filtered_df)


# In[ ]:


#Findinf, Visualizing and dropping the columns with more fraction of missing values

def view_missing_values(df_i, df_i1,preprocess_step):
    def view_missing(df, df1, thresh, thresh1):
        lst = []
        lst1 = []
        for col in df.columns:
            st.write("percentage of missing values for ",col, " ",df[col].isna().sum() / len(df) * 100)
            if ((df[col].isna().sum() / len(df) * 100) > thresh) and ((df[col].isna().sum() / len(df) * 100) <= thresh1) :
                lst.append(col)
            if (df[col].isna().sum() / len(df) * 100) > thresh1:
                lst1.append(col)

        st.write("The list of columns having fraction of missing values above the lower threshold are-", lst)
        st.write("The list of columns with more than upper threshold missing values are-", lst1)

        df = df.drop(lst1, axis=1)
        df1 = df1.drop(lst1, axis=1)

        if st.checkbox("Do you want to drop any columns above the lower threshold?"):
            st.write("The list of columns are:")
            for i, col in enumerate(lst, start=1):
                st.write(f"{i}. {col}")
            n = st.number_input("Enter the number of columns you wish to drop", min_value=0, max_value=len(lst), value=0)
            if n > 0:
                a = st.multiselect("Enter the column names you wish to drop", options=lst, default=[], max_selected=n)
                df = df.drop(a, axis=1)
                df1 = df1.drop(a, axis=1)
        return df,df1

    st.header("View missing values:")
    thresh = st.slider("Lower threshold for missing values (%)", 0, 100, 50)
    thresh1 = st.slider("Upper threshold for missing values (%)", 0, 100, 90)
    df_m,df_m1 = view_missing(df_i, df_i1, thresh, thresh1)
    st.write("The train dataframe after dropping columns is", df_m.head(20))
#     if st.button("Do you want to use this as the final processed data and proceed for step no. "+str(preprocess_step)):
#         
    return df_m,df_m1


# In[ ]:


#Imputing/ Replacing the missing values using different approaches

d = defaultdict(LabelEncoder)
# Function to replace missing values with median (if int or float) or mode (if not)
def replace_missing_values(df_i, df_i1,preprocess_step):
    df_date=df_i.select_dtypes(include='datetime64').copy()
    df_date1=df_i1.select_dtypes(include='datetime64').copy()
    df_i=df_i.select_dtypes(exclude='datetime64')
    df_i1=df_i1.select_dtypes(exclude='datetime64')
    def replace_missing_with_median_or_mode(df, df1):
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                df1[col].fillna(median, inplace=True)
            else:
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                df1[col].fillna(mode, inplace=True)
        return df,df1

    # Function to replace outliers using KNN imputer
        # Function to replace outliers using KNN imputer
    def replace_outliers_with_KNN(df,df1):
        d = defaultdict(LabelEncoder)
        d1= defaultdict(LabelEncoder)
#         df_date=df.select_dtypes(include='datetime64').copy()
#         df_date1=df1.select_dtypes(include='datetime64').copy()
#         df=df.select_dtypes(exclude='datetime64')
#         df1=df1.select_dtypes(exclude='datetime64')
        cols=df.select_dtypes(include=['object', 'category']).columns.tolist()
        df[cols]=df[cols].apply(lambda x: d[x.name].fit_transform(x))
        df1[cols]=df1[cols].apply(lambda x: d1[x.name].fit_transform(x))
        imputer = KNNImputer()
        df_imputed = imputer.fit_transform(df)
        df_imputed1=imputer.transform(df1)
        df = pd.DataFrame(df_imputed, columns=df.columns)
        df1 = pd.DataFrame(df_imputed1, columns=df1.columns)
        df[cols]=df[cols].astype(int).apply(lambda x: d[x.name].inverse_transform(x))
        df1[cols]=df1[cols].astype(int).apply(lambda x: d1[x.name].inverse_transform(x))
#         df=le.inverse_transform(df)
#         df1=le1.inverse_transform(df1)
#         df=df.join(df_date)
#         df1=df1.join(df_date1)
        return df,df1


    # Choose missing value replacement method
    st.header("Choose missing value replacement method:")
    missing_value_method = st.radio("", ["Replace with median or mode", "Replace with KNN"])
    if missing_value_method == "Replace with median or mode":
        df_m, df_m1 = replace_missing_with_median_or_mode(df_i, df_i1)
    else:
        df_m, df_m1 = replace_outliers_with_KNN(df_i, df_i1)
    df_m=df_m.join(df_date)
    df_m1=df_m1.join(df_date1)
    st.write("Cleaned train Data:", df_m.head(20))
#     if st.button("Do you want to use this as the final processed data and proceed for step no. "+str(preprocess_step)):
#         
    return df_m, df_m1


# In[ ]:


#Removing the outliers using different methods

def remove_outliers(df_i, df_i1,preprocess_step):
    df_date=df_i.select_dtypes(include='datetime64').copy()
    df_date1=df_i1.select_dtypes(include='datetime64').copy()
    df_i=df_i.select_dtypes(exclude='datetime64')
    df_i1=df_i1.select_dtypes(exclude='datetime64')
    def univariate_method(df, multiplier):
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - multiplier * std
                upper_bound = mean + multiplier * std
                df[col] = df[col].clip(lower_bound, upper_bound)
        return df

    def multivariate_method(df, n_estimators, max_samples, contamination, random_state):
        model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, random_state=random_state)
        model.fit(df)
        outliers = model.predict(df) == -1
        df = df[~outliers]
        return df

    def minkowski_error(df, p):
        distances = pdist(df.sub(df.mean()), metric='minkowski', p=p)
        pos=[]
        for i in range(len(df)):
            if distances[i]>3:
                pos.append(i)
        return df.iloc[pos,:]

    st.header("Remove outliers from data")
    method = st.selectbox("Select a method to deal with outliers:", ["Univariate Method", "Multivariate Method", "Minkowski Error"])

    if method == "Univariate Method":
        multiplier = st.slider("Enter a multiplier for the standard deviation", 0.1, 5.0, step=0.1)
        df_t = univariate_method(df_i, multiplier)
        df_t=df_t.join(df_date)
        df_t1=df_i1.copy()
        df_t1=df_t1.join(df_date1)
    elif method == "Multivariate Method":
        n_estimators = st.slider('Number of trees', 1, 100, 10)
        max_samples = st.slider('Maximum number of samples for a leaf', 1, 100, 10)
        contamination = st.slider('Percentage of outliers', 0.0, 1.0, 0.05)
        random_state = st.number_input('Random seed', value=0)
        df_t = multivariate_method(df_i, n_estimators, max_samples, contamination, random_state)
        df_t=df_t.join(df_date)
        df_t1=df_i1.copy()
        df_t1=df_t1.join(df_date1)
    elif method == "Minkowski Error":
        p = st.slider("Enter the value of p for Minkowski distance calculation", 1, 10, step=1)
        df_t = minkowski_error(df_i, p)
        df_t=df_t.join(df_date)
        df_t1=df_i1.copy()
        df_t1=df_t1.join(df_date1)

    st.write("Train Data after removing outliers:", df_t.head(20))
    st.write("Test Data:", df_t1.head(20))
    st.write("Train Dataset shape", df_t.shape, "Test Dataset shape", df_t1.shape)
#     if st.button("Do you want to use this as the final processed data and proceed for step no. "+str(preprocess_step)):
    return(df_t, df_t1)


# In[ ]:


#Encoding the categorical columns using different approaches

def encode_categorical(df_i, df_i1):
    def categorical_processing(df):
        processed_df = df.copy()
        for column in df.columns:
            if df[column].dtype == 'object':
                column_type = st.selectbox(f"Is column '{column}' nominal or ordinal?", ['n', 'o'])
                if column_type == 'n':
                    one_hot = OneHotEncoder()
                    one_hot_encoded = one_hot.fit_transform(df[[column]]).toarray()
                    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=[f"{column}_{val}" for val in one_hot.categories_[0]])
                    processed_df = processed_df.drop(column, axis=1)
                    processed_df = processed_df.join(one_hot_encoded_df)
                elif column_type == 'o':
                    label_encoding = st.selectbox(f"Do you want to use label encoding for column '{column}'?", ['y', 'n'])
                    if label_encoding == 'y':
                        label_encoder = LabelEncoder()
                        processed_df[column] = label_encoder.fit_transform(df[column])
                    else:
                        categories = df[column].unique()
                        st.write(f"Enter the numbering order for the categories in column '{column}':")
                        for i, category in enumerate(categories):
                            st.write(f"{i}. {category}")
                        numbering_order = st.text_input().split()
                        processed_df[column] = df[column].map({categories[int(i)]: i for i in numbering_order})
        return processed_df


    st.header("Encoding using using Streamlit")
    processed_df = categorical_processing(df_i)
    st.write(processed_df.head(20))
    if st.button("Do you want to use this as the final processed data and proceed"):
        return processed_df


# In[ ]:


#Data Reduction using different approaches

def data_reduction(df_i, df_i1,preprocess_step):
    df_date=df_i.select_dtypes(include='datetime64').copy()
    df_date1=df_i1.select_dtypes(include='datetime64').copy()
    df_i=df_i.select_dtypes(exclude='datetime64')
    df_i1=df_i1.select_dtypes(exclude='datetime64')
    def low_variance_filter(df, df1, threshold):
        return df.loc[:, df[df.select_dtypes(include='number').columns.tolist()].var() >= threshold], df1.loc[:, df[df.select_dtypes(include='number').columns.tolist()].var() >= threshold]

    def high_correlation_filter(df, df1, threshold):
        corr = df.corr()
        high_corr = corr.abs() > threshold
        to_drop = set()
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if high_corr.iloc[i,j]:
                    to_drop.add(j)
        return df.drop(df.columns[list(to_drop)], axis=1), df1.drop(df.columns[list(to_drop)], axis=1)

    def principle_component_analysis(df, df1, n_components):
        pca = PCA(n_components=n_components)
        df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC{}".format(i+1) for i in range(n_components)])
        df_pca1= pd.DataFrame(pca.transform(df1), columns=["PC{}".format(i+1) for i in range(n_components)])
        return df_pca, df_pca1


    st.header("Data Reduction using Streamlit")
    method = st.selectbox("Select a method for data reduction", ("Low Variance Filter", "High Correlation Filter", "Principle Component Analysis"))
    if method == "Low Variance Filter":
        threshold = st.slider("Enter a threshold for variance", 0.0, 1.0, 0.5)
        df_reduced, df_reduced1 = low_variance_filter(df_i, df_i1, threshold)
    elif method == "High Correlation Filter":
        threshold = st.slider("Enter a threshold for correlation", 0.0, 1.0, 0.8)
        df_reduced, df_reduced1 = high_correlation_filter(df_i, df_i1, threshold)
    elif method == "Principle Component Analysis":
        n_components = st.slider("Enter the number of components", 1, df_i.shape[1], df_i.shape[1]//2)
        df_reduced, df_reduced1 = principle_component_analysis(df_i, df_i1, n_components)
    df_reduced=df_reduced.join(df_date)
    df_reduced1=df_reduced1.join(df_date1)
    st.write("Train Data after reduction:", df_reduced.head(20))
    st.write("Test Data after reduction:", df_reduced1.head(20))
#     if st.button("Do you want to use this as the final processed data and proceed for step no. "+str(preprocess_step)):
    return df_reduced, df_reduced1


# In[ ]:


def convert_cols_datetime(df_i):   
    def convert_to_datetime(df, date_column, date_format):
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        return df

    st.header("Convert Date Column to Datetime Format")

    # Choose the date column
    if df_i is not None:
        date_columns = df_i.columns[df_i.dtypes == "object"]
        date_column = st.sidebar.multiselect("Choose the date column", date_columns)

        for column in date_column:
            df_tm=df_i.copy()
            # Choose the date format
            date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%Y-%d-%m", "%m-%d-%Y", "%Y/%m/%d", "%Y/%d/%m", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%d-%m %H:%M:%S", "%Y-%d-%m %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S.%f", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S.%f",  "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%d/%m %H:%M:%S", "%Y/%d/%m %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%d %H:%M", "%Y-%d-%m %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M", "%Y/%m/%d %H:%M", "%Y/%d/%m %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]
            date_format = st.sidebar.selectbox(f"Choose the date format for {column}", date_formats)

#             if st.button("Convert"):
            df_tm = convert_to_datetime(df_i, column, date_format)
    st.write("Date column converted to datetime format")
    st.dataframe(df_tm)
    return df_i

def convert_cols_datetime1(df_i, df_i1):   
    def convert_to_datetime(df, df1, date_column, date_format):
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        df1[date_column] = pd.to_datetime(df1[date_column], format=date_format)
        return df, df1

    st.header("Convert Date Column to Datetime Format")

    # Choose the date column
    if df_i is not None and df_i1 is not None:
        date_columns = df_i.columns[df_i.dtypes == "object"]
        date_column = st.sidebar.multiselect("Choose the date column", date_columns)

        for column in date_column:
            df_tm=df_i.copy()
            # Choose the date format
            date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%Y-%d-%m", "%m-%d-%Y", "%Y/%m/%d", "%Y/%d/%m", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%d-%m %H:%M:%S", "%Y-%d-%m %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S.%f", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S.%f",  "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%d/%m %H:%M:%S", "%Y/%d/%m %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%d %H:%M", "%Y-%d-%m %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M", "%Y/%m/%d %H:%M", "%Y/%d/%m %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]
            date_format = st.sidebar.selectbox(f"Choose the date format for {column}", date_formats)

#             if st.button("Convert"):
            df_tm, df_tm1 = convert_to_datetime(df_i, df_i1, column, date_format)
    st.write("Date column converted to datetime format")
    st.dataframe(df_tm)
    st.dataframe(df_tm1)
    return df_tm, df_tm1


# In[ ]:


def split_train_test(data,split_ratio):
    '''
    Splits the input dataset into train and test set according to provided split ratio. 
    
             Parameters:
                     data (pandas)      : dataframe
                     split_ratio (float): ratio in which entire data getting splitted i.e. (train:test) ratio
             
             Returns:
                     train (pandas)  : training dataframe
                     test (pandas)   : testing dataframe
    '''
    train_size = int(len(data)*split_ratio) 
    # test_size = len(data) - train_size
    train, test = data.iloc[0:train_size],data.iloc[train_size:len(data)]
    print("Train Shape: {0}  Test Shape: {1}".format(train.shape, test.shape))
    
    return train, test


# In[ ]:
def seasonal(current_df, current_df1,date_ano):
#     current_df_sea=pd.concat([current_df, current_df1])
    current_df=current_df.reset_index()
    current_df1=current_df1.reset_index()
    for i in current_df.select_dtypes(include='number').columns:
        st.subheader("Time series deseasonalizing for column {}".format(i))
        sea_resp=st.radio("Want to deseasonalize column {}?".format(i), ['n','y'])
        if sea_resp=='y':
            current_df_sea=current_df[[date_ano, i]]
            current_df_sea.columns=['ds','y']
            current_df_sea1=current_df1[[date_ano, i]]
            current_df_sea1.columns=['ds','y']
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(10,6))
            sns.lineplot(x='ds', y='y', data=current_df_sea)
            plt.title(f"Distribution of train data for {i}")
            plt.xlabel(current_df_sea.columns[0])
            plt.ylabel(current_df_sea.columns[1])
            st.pyplot()
            plt.figure(figsize=(10,6))
            sns.lineplot(x='ds', y='y', data=current_df_sea1)
            plt.title(f"Distribution of test data for {i}")
            plt.xlabel(current_df_sea1.columns[0])
            plt.ylabel(current_df_sea1.columns[1])
            st.pyplot()
            weekly_op=st.radio("Does your data have weekly seasonlaity for column {}?".format(i), ['n','y'])
            if weekly_op=='y':
                weekly_seasonality1=int(st.text_input("Please select the fourier order for weekly seasonality for column {}".format(i)))
            elif weekly_op=='n':
                weekly_seasonality1=False

            monthly_op=st.radio("Does your data have monthly seasonlaity for column {}?".format(i), ['n','y'])
            if monthly_op=='y':
                monthly_seasonality1=int(st.text_input("Please select the fourier order for monthly seasonality for column {}".format(i)))
                name_monthly='monthly'
                period_monthly=30.5


            yearly_op=st.radio("Does your data have yearly seasonlaity for column {}?".format(i), ['n','y'])
            if yearly_op=='y':
                yearly_seasonality1=int(st.text_input("Please select the fourier order for yearly seasonality for column {}".format(i)))
            elif yearly_op=='n':
                yearly_seasonality1=False
                
            daily_op=st.radio("Does your data have daily seasonlaity for column {}?".format(i), ['n','y'])
            if daily_op=='y':
                daily_seasonality1=int(st.text_input("Please select the fourier order for daily seasonality for column {}".format(i)))
            elif daily_op=='n':
                daily_seasonality1=False
                
            detrend_op=st.radio("Do you want to detrend column {}?".format(i), ['n','y'])

    #         custom_op=st.radio(r"Does your data have any othe custom seasonlaity for column {i}?", ['n','y'])
    #         if custom_op=='y':
    #             custom_no=st.textbox(r"Please enter the number of custom seasonalities for column {i}")
    #             name_custom_sea={}
    #             custom_period={}
    #             custom_fourier={}
    #             for j in range(1,custom_no+1):
    #                 name_custom_sea[j]=st.textbox(r"Please enter the name for custom seasonality no {j} for column {i}")
    #                 custom_period[j]=st.textbox(r"Please enter the period for custom seasonality no {j} for column {i}")
    #                 custom_fourier[j]=st.slider("Please select the fourier order for custom seasonality {j} for column {i}"5,20,12)
            growth1=st.text_input("Please select the growth term for the column {}, options are- ['linear', 'logistic', 'flat'], default is 'linear'".format(i))
            seasonality_prior_scale1=float(st.text_input("Please select the seasonality prior scale for column {}, default is 10".format(i)))
            seasonality_mode1=st.text_input("Please select the seasonality mode for column {}, options are- ['additive', 'multiplicative'], default is 'additive'".format(i))
            changepoint_prior_scale1=float(st.text_input("Please select the changepoint prior scale for column {}, default is 0.05".format(i)))
            interval_width1=float(st.text_input("Please select the interval width for column {}, default is 0.8".format(i)))
            changepoint_range1=float(st.text_input("Please select the changepoint range interval for column {}, default is 0.8".format(i)))
            if monthly_op=='y':
                m = Prophet(growth=growth1, seasonality_prior_scale=seasonality_prior_scale1, seasonality_mode=seasonality_mode1, daily_seasonality=daily_seasonality1, weekly_seasonality=weekly_seasonality1, yearly_seasonality=yearly_seasonality1, changepoint_prior_scale=changepoint_prior_scale1, interval_width=interval_width1, changepoint_range=changepoint_range1).add_seasonality(name=name_monthly, period=period_monthly, fourier_order=monthly_seasonality1)
                if seasonality_mode1=='additive':
                    forecast = m.fit(current_df_sea).predict(current_df_sea.drop(['y'],axis = 1))
                    st.write(forecast.columns)
                    st.write("The mean absolute error is", mean_absolute_error(forecast['yhat'],current_df[i]))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df[i]=current_df[i]-forecast[param]
                    current_df[i]=current_df[i]-forecast[name_monthly]
                    if detrend_op=='y':
                        current_df[i]=current_df[i]-forecast['trend']
                    forecast1=m.predict(current_df_sea1.drop(['y'],axis = 1))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df1[i]=current_df1[i]-forecast1[param]
                    current_df1[i]=current_df1[i]-forecast1[name_monthly]
                    if detrend_op=='y':
                        current_df1[i]=current_df1[i]-forecast1['trend']
                elif seasonality_mode1=='multiplicative':
                    forecast = m.fit(current_df_sea).predict(current_df_sea.drop(['y'],axis = 1))
                    st.write(forecast.columns)
                    st.write("The mean absolute error is", mean_absolute_error(forecast['yhat'],current_df[i]))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df[i]=current_df[i]/((forecast[param]))
                    current_df[i]=current_df[i]/(forecast.name_monthly)
                    if detrend_op=='y':
                        current_df[i]=current_df[i]-forecast['trend']
                    forecast1=m.predict(current_df_sea1.drop(['y'],axis = 1))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df1[i]=current_df1[i]/((forecast1[param]))
                    current_df1[i]=current_df1[i]/(forecast1.name_monthly)
                    if detrend_op=='y':
                        current_df1[i]=current_df1[i]-forecast1['trend']
            elif monthly_op=='n':
                m = Prophet(growth=growth1, seasonality_prior_scale=seasonality_prior_scale1, seasonality_mode=seasonality_mode1, yearly_seasonality=yearly_seasonality1, weekly_seasonality=weekly_seasonality1, daily_seasonality=daily_seasonality1, changepoint_prior_scale=changepoint_prior_scale1, interval_width=interval_width1, changepoint_range=changepoint_range1)
                if seasonality_mode1=='additive':
                    forecast = m.fit(current_df_sea).predict(current_df_sea.drop(['y'],axis = 1))
                    st.write(forecast.columns)
                    st.write("The mean absolute error is", mean_absolute_error(forecast['yhat'],current_df[i]))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df[i]=current_df[i]-forecast[param]
                    if detrend_op=='y':
                        current_df[i]=current_df[i]-forecast['trend']
                    forecast1=m.predict(current_df_sea1.drop(['y'],axis = 1))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df1[i]=current_df1[i]-forecast1[param]
                    if detrend_op=='y':
                        current_df1[i]=current_df1[i]-forecast1['trend']
                elif seasonality_mode1=='multiplicative':
                    forecast = m.fit(current_df_sea).predict(current_df_sea.drop(['y'],axis = 1))
                    st.write(forecast.columns)
                    st.write("The mean absolute error is", mean_absolute_error(forecast['yhat'],current_df[i]))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df[i]=current_df[i]/((forecast[param]))
                    if detrend_op=='y':
                        current_df[i]=current_df[i]-forecast['trend']
                    forecast1=m.predict(current_df_sea1.drop(['y'],axis = 1))
                    for param in ['weekly', 'yearly', 'daily']:
                        if param in forecast.columns:
                            current_df1[i]=current_df1[i]/((forecast1[param]))
                    if detrend_op=='y':
                        current_df1[i]=current_df1[i]-forecast1['trend']

            plt.figure(figsize=(10,6))
            sns.lineplot(x=date_ano, y=i, data=current_df)
            plt.title(f"Distribution of train data for {i} after detrending and deseasonalizing")
            plt.xlabel(current_df.columns[0])
            plt.ylabel(current_df.columns[1])
            st.pyplot()
            plt.figure(figsize=(10,6))
            sns.lineplot(x=date_ano, y=i, data=current_df1)
            plt.title(f"Distribution of test data for {i} after detrending and deseasonalizing")
            plt.xlabel(current_df1.columns[0])
            plt.ylabel(current_df1.columns[1])
            st.pyplot()            
        else:
            pass

    return current_df, current_df1

def scale(current_df, current_df1, current_df_tm, current_df_tv):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(current_df)
    X_test = scaler.transform(current_df1)
    X_model_train= scaler.transform(current_df_tm)
    X_model_val= scaler.transform(current_df_tv)
    return X_train, X_test, X_model_train, X_model_val

def create_sequences(current_df, current_df1, current_df_tm, current_df_tv, time_steps):
    Xs= []
    for i in range(len(current_df)-time_steps+1):
        Xs.append(current_df[i:(i+time_steps)])
    
    Xs1= []
    for i in range(len(current_df1)-time_steps+1):
        Xs1.append(current_df1[i:(i+time_steps)])
        
    Xs_tm= []
    for i in range(len(current_df_tm)-time_steps+1):
        Xs_tm.append(current_df_tm[i:(i+time_steps)])
        
    Xs_tv= []
    for i in range(len(current_df_tv)-time_steps+1):
        Xs_tv.append(current_df_tv[i:(i+time_steps)])




    return np.array(Xs), np.array(Xs1), np.array(Xs_tm), np.array(Xs_tv)

def create_sequences_mask(current_df, current_df1, time_steps):
    Xs= []
    for i in range(len(current_df)-time_steps+1):
        Xs.append(current_df[i:(i+time_steps)])
    
    Xs1= []
    for i in range(len(current_df1)-time_steps+1):
        Xs1.append(current_df1[i:(i+time_steps)])


    return np.array(Xs), np.array(Xs1)

def batch_generator(data, batch_size):
    # Define a generator function to yield batches of sequences
    def sequence_generator(data, batch_size):
        num_inputs, seq_len, num_features = data.shape
        num_batches = num_inputs // batch_size

        for i in range(num_batches):
            batch_data = data[i * batch_size : (i + 1) * batch_size]
            yield batch_data

    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: sequence_generator(data, batch_size),
        output_signature=tf.TensorSpec(shape=(None, data.shape[1], data.shape[2]), dtype=tf.float32)
    )
    
    lst=[]
    for batch in dataset:
        lst.append(batch)
    return lst

def batch_generator_mask(data, batch_size):
    # Define a generator function to yield batches of sequences
    def sequence_generator(data, batch_size):
        num_inputs, seq_len = data.shape
        num_batches = num_inputs // batch_size

        for i in range(num_batches):
            batch_data = data[i * batch_size : (i + 1) * batch_size]
            yield batch_data

    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: sequence_generator(data, batch_size),
        output_signature=tf.TensorSpec(shape=(None, data.shape[1]), dtype=tf.float32)
    )
    
    lst=[]
    for batch in dataset:
        lst.append(tf.reshape(batch,(batch.shape[0],batch.shape[1],1)))
    return lst

#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_

class Encoder(tf.keras.Model):
    def __init__(self, n_hidden, neurons, latent_dim, input_shape):
        super(Encoder, self).__init__()
        self.n_hidden = n_hidden
        self.neurons = neurons
        self.latent_dim = latent_dim
        self.input_shape_1 = input_shape
        self.encoder_layers = self.build_encoder_layers()

    def build_encoder_layers(self):
        st.write(self.n_hidden,self.neurons,self.latent_dim,self.input_shape_1)
        encoder_layers = []
        for i in range(self.n_hidden):
            if i == 0:
                encoder_layers.extend([
                    layers.Bidirectional(layers.LSTM(self.neurons, input_shape=self.input_shape_1, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), return_sequences=True)),
                    layers.BatchNormalization()
                ])
            elif i == self.n_hidden - 1:
                encoder_layers.append(layers.Bidirectional(layers.LSTM(self.latent_dim, activation='relu',kernel_initializer = tf.keras.initializers.HeNormal(), return_sequences=False)))
            else:
                encoder_layers.extend([
                    layers.Bidirectional(layers.LSTM(int(self.neurons - 0.14 * i * self.neurons), activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), return_sequences=True)),layers.BatchNormalization()
                ])
        encoder_layers.append(layers.RepeatVector(self.input_shape_1[0]))
        return encoder_layers

    def call(self, w):
        x = w
        for layer in self.encoder_layers:
            x = layer(x)
#         st.write(x.shape,self.latent_dim)
        return x

    
class Decoder(tf.keras.Model):
    def __init__(self, n_hidden, neurons, latent_dim, input_shape):
        super(Decoder, self).__init__()
        self.n_hidden = n_hidden
        self.neurons = neurons
        self.latent_dim = latent_dim
        self.input_shape_1 = input_shape
        self.decoder_layers = self.build_decoder_layers()

    def build_decoder_layers(self):
        decoder_layers = []
#         st.write(self.n_hidden)
        for i in range(self.n_hidden):
#             st.write(i)
            if i == 0:
                decoder_layers.append(layers.Bidirectional(layers.LSTM(self.latent_dim, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), return_sequences=True)))
            else:
                decoder_layers.append(layers.Bidirectional(layers.LSTM(int(self.neurons - 0.14 * (self.n_hidden - 1 - i) * self.neurons), activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), return_sequences=True)))
                if i % 2 == 0:
                    decoder_layers.append(layers.BatchNormalization())
        decoder_layers.append(layers.TimeDistributed(layers.Dense(self.input_shape_1[1])))
        return decoder_layers

    def call(self, w):
#         st.write(w.shape)
        x = w
        for layer in self.decoder_layers:
#             hidden_model = Model(inputs=w, outputs=hidden_layer.output)
#             hidden_output = hidden_model.predict(x_test)
            x = layer(x)
        return x

    
class AutoEncoderModel(tf.keras.Model):
    def __init__(self, n_hidden, neurons, latent_dim, input_shape):
        super(AutoEncoderModel, self).__init__()
        self.encoder = Encoder(n_hidden, neurons, latent_dim, input_shape)
        self.decoder1 = Decoder(n_hidden, neurons, latent_dim, input_shape)
        self.decoder2 = Decoder(n_hidden, neurons, latent_dim, input_shape)
        
  
    def training_step(self, batch, sample_weights,n):
        #Phase-1
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        
        #Phase-2
        w3 = self.decoder2(self.encoder(w1))
        
        loss1 = 1/n * tf.reduce_mean(tf.square(tf.math.multiply(batch - w1,sample_weights))) + (1 - 1/n) * tf.reduce_mean(tf.square(tf.math.multiply(batch - w3,sample_weights)))
        loss2 = 1/n * tf.reduce_mean(tf.square(tf.math.multiply(batch - w2,sample_weights))) - (1 - 1/n) * tf.reduce_mean(tf.square(tf.math.multiply(batch - w3,sample_weights)))
        
        return loss1, loss2
    
    def validation_step(self, batch, sample_weights, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n * tf.reduce_mean(tf.square(tf.math.multiply(batch - w1, sample_weights))) + (1 - 1/n) * tf.reduce_mean(tf.square(tf.math.multiply(batch - w3, sample_weights)))
        loss2 = 1/n * tf.reduce_mean(tf.square(tf.math.multiply(batch - w2, sample_weights))) - (1 - 1/n) * tf.reduce_mean(tf.square(tf.math.multiply(batch - w3, sample_weights)))
        st.write(loss1,loss2)
        return {'val_loss1': loss1, 'val_loss2': loss2}
        
    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = tf.reduce_mean(batch_losses1)
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = tf.reduce_mean(batch_losses2)
        return {'val_loss1': epoch_loss1.numpy(), 'val_loss2': epoch_loss2.numpy()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss1: {result['val_loss1']:.4f}, val_loss2: {result['val_loss2']:.4f}")
        
def evaluate(model, val_dataset, sample_weights_val, n):
    outputs = [model.validation_step(val_dataset[i],sample_weights_val[i],n) for i in range(len(val_dataset))]
    return model.validation_epoch_end(outputs)

def val_testing(model, test_dataset, alpha=.5, beta=.5):
    results = []
    for batch in test_dataset:
        batch = tf.convert_to_tensor(batch)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * tf.reduce_mean(tf.square(batch - w1)) + beta * tf.reduce_mean(tf.square(batch - w2)))
        
    return tf.reduce_mean(results)


def training(epochs, model, train_dataset, val_dataset, sample_weights_train, sample_weights_val, batch_size, lr, seq_len, opt_func=tf.optimizers.Adam):
    history = []
    optimizer1 = opt_func(learning_rate=lr)
    optimizer2 = opt_func(learning_rate=lr)
    train_dataset, val_dataset, sample_weights_train, sample_weights_val= batch_generator(train_dataset,batch_size), batch_generator(val_dataset,batch_size), batch_generator_mask(sample_weights_train,batch_size), batch_generator_mask(sample_weights_val,batch_size)
    
    for epoch in range(epochs):
#         for batch in train_dataset:
        for i in range(len(train_dataset)):
            with tf.GradientTape(persistent=True) as tape:
                batch = train_dataset[i]
                sample_weight=sample_weights_train[i]
                batch = tf.convert_to_tensor(batch)
                sample_weight=tf.convert_to_tensor(sample_weight)
                
                # Train AE1
                loss1, loss2 = model.training_step(batch,sample_weight,epoch + 1)
                st.write("epoch",epoch,"batch no",i,loss1,loss2)
                #Propogating-Loss1
                gradients1 = tape.gradient(loss1, model.encoder.trainable_variables + model.decoder1.trainable_variables)
                optimizer1.apply_gradients(zip(gradients1, model.encoder.trainable_variables + model.decoder1.trainable_variables))
                
                # Train AE2
                loss1, loss2 = model.training_step(batch,sample_weight,epoch + 1)
                st.write("epoch",epoch,"batch no",i,loss1,loss2)
                #Propogating-Loss2
                gradients2 = tape.gradient(loss2, model.encoder.trainable_variables + model.decoder2.trainable_variables)
                optimizer2.apply_gradients(zip(gradients2, model.encoder.trainable_variables + model.decoder2.trainable_variables))
        
        result = evaluate(model, val_dataset, sample_weights_val, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    val_loss_tot= val_testing(model, val_dataset,0.5,0.5)
    return history,val_loss_tot

#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_

def create_model(input_shape, n_hidden, neurons, loss, epochs, batch_size, metrics, optimizer, latent_dim):
    model = keras.Sequential()
    for i in range(n_hidden):
        if i == 0:
            model.add(layers.Bidirectional(layers.LSTM(neurons, input_shape=input_shape, activation='relu', return_sequences=True)))
            model.add(layers.BatchNormalization())
        elif i == n_hidden - 1:
            model.add(layers.Bidirectional(layers.LSTM(latent_dim,  activation='relu',return_sequences=False)))
        else:
            model.add(layers.Bidirectional(layers.LSTM(int(neurons-0.14*i*neurons),  activation='relu', return_sequences=True)))
    model.add(layers.RepeatVector(input_shape[0]))
    for i in range(n_hidden):
        if i == 0:
            model.add(layers.Bidirectional(layers.LSTM(latent_dim,  activation='relu',return_sequences=True)))
        else:
           
        
            model.add(layers.Bidirectional(layers.LSTM(int(neurons-0.14*(n_hidden-1-i)*neurons),  activation='relu',return_sequences=True)))
            if i%2==0:
                model.add(layers.BatchNormalization())
    model.add(layers.TimeDistributed(layers.Dense(input_shape[1])))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_


def get_loss(params, X_train, X_val, X_mask_model_tm, X_mask_model_tv):
    model = AutoEncoderModel(int(params["n_hidden"]),int(params["neurons"]),int(params["latent_dim"]),(X_train.shape[1], X_train.shape[2]))
#     model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]),
#                         n_hidden=int(params["n_hidden"]),
#                         neurons=int(params["neurons"]),
#                         loss=params["loss"],
#                         epochs=int(params["epochs"]),
#                         batch_size=int(params["batch_size"]),
#                         metrics=["mse"],
#                         optimizer=Adam(lr=params["lr"]),
#                         latent_dim=int(params["latent_dim"]))
#     history = model.fit(X_train, X_train,
#                         epochs=int(params["epochs"]),
#                         batch_size=int(params["batch_size"]),
#                         sample_weight=X_mask_model_tm,
#                         validation_data=(X_val, X_val,X_mask_model_tv), shuffle=False, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',patience=12)])

    history, val_loss= training(5, model, X_train, X_val, X_mask_model_tm, X_mask_model_tv, int(params["batch_size"]), float(params["lr"]), tf.optimizers.Adam)
#     val_loss = np.amin(history.history["val_loss"])
    return {'loss': val_loss, 'model': model, 'params': params, 'status': STATUS_OK}

@st.cache_resource
def hyperparameter_tuning(X_train, X_val, X_mask_model_tm, X_mask_model_tv):
    trials = Trials()
    space = {
        "n_hidden": hp.quniform("n_hidden", 2, 3, 1),
        "neurons": hp.quniform("neurons", 16, 28, 2),
#         "loss": hp.choice("loss", ["mse", "mae"]),
#         "epochs": hp.quniform("epochs", 100, 500, 1),
#         "batch_size": hp.quniform("batch_size", 32, 256, 32),
        "batch_size": hp.choice("batch_size",[32,64,128,256]),
        "lr": hp.loguniform("lr", -5, -4),
        "latent_dim": hp.quniform("latent_dim", 4, 6, 1)
    }
    best = fmin(fn=lambda params: get_loss(params, X_train, X_val, X_mask_model_tm, X_mask_model_tv),
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=3,
                verbose=1)
    best_params={"n_hidden": int(best["n_hidden"]), "neurons":int(best["neurons"]), "batch_size":int(best["batch_size"]), "lr":float(best["lr"]), "latent_dim":int(best["latent_dim"])}
    st.write("Best hyperparameters:",best_params)
    return trials.results[np.argmin([r['loss'] for r in trials.results])]['model']


def detect_anomaly(model,x_train,df_train, x_test, df_test,cols,t,prefix='',thresh="99",top_percent=0.05):
    fig_dict = defaultdict(dict)
    anomaly_dict = defaultdict(dict)
    scored_dict = defaultdict(dict)
    
    x_train = tf.convert_to_tensor(x_train)
    X_pred_train_1_t = model.decoder1(model.encoder(x_train))
    X_pred_train_1=tf.reshape(X_pred_train_1_t,(X_pred_train_1_t.shape[0]*X_pred_train_1_t.shape[1], X_pred_train_1_t.shape[2]))
    Xs=[]
    for k in range(len(df_train)-t+1):
        for l in range(k,k+t):
            Xs.append(l)

    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=X_pred_train_1
    vals=df_temp.groupby(['idx'])[cols].mean().values
      
    X_pred_train_1 = pd.DataFrame(vals, columns = cols)         
    X_pred_train_1.index = df_train.index
                            
                            
    X_pred_train_2_t = model.decoder2(model.encoder(X_pred_train_1_t))
    X_pred_train_2=tf.reshape(X_pred_train_2_t,(X_pred_train_2_t.shape[0]*X_pred_train_2_t.shape[1], X_pred_train_2_t.shape[2]))
    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=X_pred_train_2
    vals=df_temp.groupby(['idx'])[cols].mean().values
    X_pred_train_2 = pd.DataFrame(vals, columns = cols)         
    X_pred_train_2.index = df_train.index
                            
    scored_train = pd.DataFrame(index=df_train.index)
#############################################################################################################################    
    
    x_train=tf.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2]))
    Xs=[]
    for k in range(len(df_train)-t+1):
        for l in range(k,k+t):
            Xs.append(l)

    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=x_train
    vals=df_temp.groupby(['idx'])[cols].mean().values
    
    Xtrain = vals
    scored_train['Loss_mae'] = np.mean(np.abs((0.5*X_pred_train_1.values+0.5*X_pred_train_2.values) - Xtrain), axis=1)
    
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored_train['Loss_mae'], bins=20, kde=True, color='blue');
    plt.xlim()
    
    thresh_list = []
#     for i in dens_percent:
    thresh_mae = np.percentile(scored_train.Loss_mae, float(thresh))
    thresh_list.append(thresh_mae)
    print("Detecting anomalies for following thresholds: ", thresh_list)
############################################################################################################################################   
    x_test = tf.convert_to_tensor(x_test)
    X_pred_test_1_t = model.decoder1(model.encoder(x_test))
    X_pred_test_1=tf.reshape(X_pred_test_1_t,(X_pred_test_1_t.shape[0]*X_pred_test_1_t.shape[1], X_pred_test_1_t.shape[2]))
    Xs=[]
    for k in range(len(df_test)-t+1):
        for l in range(k,k+t):
            Xs.append(l)

    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=X_pred_test_1
    vals=df_temp.groupby(['idx'])[cols].mean().values
    X_pred_test_1 = pd.DataFrame(vals, columns = [cols])
    X_pred_test_1.index = df_test.index
                            
    
    X_pred_test_2_t = model.decoder2(model.encoder(X_pred_test_1_t))
    X_pred_test_2=tf.reshape(X_pred_test_2_t,(X_pred_test_2_t.shape[0]*X_pred_test_2_t.shape[1], X_pred_test_2_t.shape[2]))
    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=X_pred_test_2
    vals=df_temp.groupby(['idx'])[cols].mean().values
    X_pred_test_2 = pd.DataFrame(vals, columns = [cols])
    X_pred_test_2.index = df_test.index
                            
    scored_test = pd.DataFrame(index=df_test.index)
####################################################################################################################################    
    
    x_test=tf.reshape(x_test,(x_test.shape[0]*x_test.shape[1], x_test.shape[2]))
    Xs=[]
    for k in range(len(df_test)-t+1):
        for l in range(k,k+t):
            Xs.append(l)

    df_temp= pd.DataFrame(Xs, columns = ["idx"])
    df_temp[cols]=x_test
    vals=df_temp.groupby(['idx'])[cols].mean().values
 
    
    Xtest = vals
    scored_test['Loss_mae'] = np.mean(np.abs((0.5*X_pred_test_1.values+0.5*X_pred_test_2.values) - Xtest), axis=1)
    dense_percent=list()
    dense_percent.append(float(thresh))
    for (thresh,dense) in zip(thresh_list,dense_percent):
        print("THRESH: ", thresh)
        scored_test['Threshold'] = thresh
        scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
    
        scored_train['Threshold'] = thresh
        scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
        scored = pd.concat([scored_train, scored_test])
        st.write(scored)     
        anomaly = scored[scored['Anomaly']==True]
        anomaly_desc = anomaly.sort_values(['Loss_mae'], ascending = False)
        
        st.write("Total Number of Anomalies Detected:",anomaly.shape[0])
        st.write("Percentage of Anomaly vs Input Data is {:.3f} % \n".format(scored['Anomaly'].value_counts(True)[1]*100))
        st.write("The anomalies are -")
        st.table(anomaly)
        st.download_button(label = "Download whole data", data = convert_df(anomaly), 
                                       file_name ="Anomaly_Data.csv", mime='text/csv')
        
#         n = anomaly.shape[0] 
# #         toWriteCSV(anomaly, prefix+'_Anomalies_'+str(dense)+'.csv')
       
#         top_ano = anomaly_desc.head(round(n*top_percent))
#         top_ano['index'] = top_ano.index
#         top_ano.reset_index(inplace=True)
        if len(anomaly_desc)>=1:
            fig = scored.plot(logy=True, figsize=(16,9), ylim=[0,0.8], color=['blue','red'])
        else:
            fig = scored['Threshold'].plot(figsize=(16,9), ylim=[0,0.8], color=['blue'])
#         plt.ylim(0,0.3)
        plt.title('Anomalies over Loss-Threshold Plot', fontsize=16)
        
#         # plt_scatter = plt_anomaly+ '_' + str(round(thresh,3))+ '.jpg' 
#         plt_scatter = plt_anomaly+ '_' + str(dense)+ '.jpg' 
        
#         if n<=20:
#             print("Detected anomalies are: \n \n", anomaly)
#             plt.scatter(anomaly.index, anomaly['Loss_mae'], c = 'red')
# #             plt.savefig(path.join(plots_path, plt_scatter))
#             print("\n" + plt_scatter + " saved successfully in requested directory. \n")
#             plt.show()
            
#         else:
#             print("Top 5% Detected anomalies are: \n", top_ano)
# #             plt.scatter(top_ano.index, top_ano['Loss_mae'], c = 'red')
#             plt.scatter(anomaly.index, anomaly['Loss_mae'], c = 'red')
# #             plt.savefig(path.join(plots_path, plt_scatter))
#             print("\n" + plt_scatter + " saved successfully in requested directory. \n")
#             plt.show()
            
        fig = fig.get_figure()
        st.write("The anomalies plot is -")
        st.pyplot(fig)
        img = io.BytesIO()
        pl=fig
        pl.savefig(img, format = 'png')
        st.download_button(label="Download image", data = img, file_name= "Anomaly_image.png", mime="image/png")
# To avoid the error of fig.savefig() is not possible.
        
        fig_dict[str(dense)] = fig
        anomaly_dict[str(dense)] = anomaly
        scored_dict[str(dense)] = scored
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def preprocessing(current_df, current_df1,history):
#     st.empty()
#     st.write("Preprocess initial:", current_df.shape)
#     st.write("Test Data:", current_df1.shape)
    i=1
    if current_df is not None and current_df1 is not None:
#         global current_df, current_df1, history
#         history = []
        st.write("Pre-processing function list  \n1.View missing values  \n2.Replace missing values  \n3.Remove Outliers  \n4.Data Reduction")
        str1=st.text_input("Please Enter the order of execution","E.g. 3 2 1 4")
        order=str1.split(' ')
        #st.write(order)
        func_name=[]
        for i in order:
            if(i=='1'):
                func_name.append("View missing values");
            elif(i=='2'):
                func_name.append("Replace missing values");
            elif(i=='3'):
                func_name.append("Remove Outliers");
            elif(i=='4'):
                func_name.append("Data Reduction");
        st.write(func_name)
        preprocess_step=0
        
        for func in func_name:
            preprocess_step+=1
            st1=st.empty()
            st.write("Preprocess initial:", current_df.shape)
            st.write("Test Data:", current_df1.shape)
#             st.sidebar.subheader("Choose each function one at a time, in the order of your preprocessing")
# #             func = st.sidebar.multiselect("Select the preprocessing function no {}".format(i), ["please select","View missing values", "Replace missing values", "Remove Outliers", "Data Reduction", "Revert the Previous Action"],key="var"+str(i))
            
            
            if func not in ["please select"]:
                if func == "View missing values":
                    new_df, new_df1 = view_missing_values(current_df, current_df1,preprocess_step)
                    history.append((current_df, current_df1))
                    current_df = new_df
                    current_df1=new_df1
                    st.write(" Train Dataset Shape:", current_df.shape)
                    st.write(" Test Dataset Shape:", current_df1.shape)
        #             i+=1
        #             if st.sidebar.button("Do you want to continue preprocessing?",key="cont"+str(i)):
        #                 preprocessing(current_df, current_df1, history,i)
                elif func == "Replace missing values":
                    new_df, new_df1 = replace_missing_values(current_df, current_df1,preprocess_step)
                    history.append((current_df, current_df1))
                    current_df = new_df
                    current_df1=new_df1
                    st.write("Train Dataset Shape:", current_df.shape)
                    st.write("Test Dataset Shape:", current_df1.shape)
        #             i+=1
        #             if st.sidebar.button("Do you want to continue preprocessing?", key="cont"+str(i)):
        #                 preprocessing(current_df, current_df1, history,i)
                elif func == "Remove Outliers":
                    new_df, new_df1 = remove_outliers(current_df, current_df1,preprocess_step)
                    history.append((current_df, current_df1))
                    current_df = new_df
                    current_df1=new_df1
                    st.write("Train Dataset Shape:", current_df.shape)
                    st.write("Test Dataset Shape:", current_df1.shape)
        #             i+=1
        #             if st.sidebar.button("Do you want to continue preprocessing?", key="cont"+str(i)):
        #                 preprocessing(current_df, current_df1, history,i)
        #         elif func == "Encoding Categorical Data":
        #             new_df = encode_categorical(current_df)
        #             history.append(current_df)
        #             current_df = new_df
        #             st.write("Dataset Shape:", current_df.shape)
        # #                 if st.button("Back to home"):
        # #                     main()
                elif func == "Data Reduction":
                    new_df, new_df1 = data_reduction(current_df, current_df1,preprocess_step)
                    history.append((current_df, current_df1))
                    current_df = new_df
                    current_df1=new_df1
                    st.write("Train Dataset Shape:", current_df.shape)
                    st.write("Test Dataset Shape:", current_df.shape)
        st.write("success");
        return current_df,current_df1,history
                    
        #             i+=1
        #             if st.sidebar.button("Do you want to continue preprocessing?", key="cont"+str(i)):
        #                 preprocessing(current_df, current_df1, history,i)
#                 elif func=="Revert The Previous Action":
#                     if st.radio("Are you sure you want to revert the previous Action", ['n','y'])=='y':
#                         if len(history)>=1:
#                             del history[-1]
#                             current_df=history[-1][0].copy()
#                             current_df1=history[-1][1].copy()
#                             st.write("Previous dataframe has been restored and is the current dataframe. The top 20 rows of train data are", current_df.head(20), "The top 20 rows of test data are", current_df1.head(20))
#                             st.write("Train Dataset Shape:", current_df.shape, "Test Dataset shape", current_df1.shape(20))
#                         else:
#                             print("Sorry, there is no history of operations to revert")
#                 response_proceed= st.sidebar.radio('Are you sure, you want to end preprocessing with this step no {}? '.format(i), ['n', 'y'])
#                 if response_proceed=='y':
#                     return current_df, current_df1, history
#                 else:
#                     i=i+1
#                     continue
#             else:
#                 break
#             if (response_proceed=='y'):                    
#                 return current_df, current_df1, history                
#             else:                    
#                 done=st.sidebar.button("Want to continue preprocessing step no {}".format(i+1))
 

        

    else:
        st.write("Either of train or test samples have no samples, please check!")
        
        
def look_data(current_df, current_df1):
    if current_df is not None and current_df1 is not None:
#         current_df = df.copy()
#         current_df1=df1.copy()
        st.sidebar.header("Hello")
        st.sidebar.subheader("Quick look at your data")
        op = st.sidebar.selectbox("View your data", ["please select","Describe data", "Binning the columns into category types", "View Data"])

        if op == "Describe data":
            st.write("The description for train data")
            describe_data(current_df)
            st.write("The description for test data")
            describe_data(current_df1)
#             if st.button("Back to home"):
#                 main()
        elif op == "Binning the columns into category types":
            st.write("The category types for train data")
            check_column_category(current_df)
            st.write("The category types for test data")
            check_column_category(current_df1)
#             if st.button("Back to home"):
#                 main()
        elif op == "View Data":
            st.write("View train data")
            view_data(current_df)
            st.write("View test data")
            view_data(current_df1)
#             if st.button("Back to home"):
#                 main()
    else:
        st.write("Either of train or test data have no samples, please check!")    
                

def main():
    st.title("Data Exploration and Preparation Tool")
    st.write("This tool provides several functionalities to explore and prepare a data set.")
    no_files=st.sidebar.radio("Please select your input file structure",["Have train and test data in a single file","Have train and test data in separate files"])
    if no_files=="Have train and test data in a single file":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
            st.write("Dataset Shape:", df.shape)
            st.write(df)
            if df is not None:    
                dtt=st.sidebar.selectbox("Does your data have datetime columns", ['n', 'y'])
                if dtt=='y':
                    df=convert_cols_datetime(df)
                change_type=st.sidebar.radio("Do you want to change the datatype of any column in the dataset", ['n', 'y'])
                if change_type=='y':
                    df=change_dtype(df)
                res=st.sidebar.selectbox("Do you want to take your analysis to a granular level based on categories", ['please select','n', 'y'])
                if res=='y':
                    n = st.sidebar.number_input('How many level types do you have ?',min_value = 1, value = 1)
                    if n>=1: # select the categorical variables only if you have n>=1
                        var = st.sidebar.multiselect('Please select the level columns',df.select_dtypes(exclude=['number','datetime64']).columns.tolist(), max_selections = n)
                        df['key'] = ''
                        level_cols=[]
                        for i in var:
                            level_cols.append(i)
                            df['key'] =  df[i].astype(str) + '-' + df['key']
                        level =  st.selectbox('Select the category level to preprocess',df['key'].unique().tolist())
                        if level in df['key'].unique().tolist():
                            if df[df['key']==level] is not None:
        #                         df_level=df.select_dtypes(include='number')
                                df_level=df[df['key']==level]
                                df_level=df_level.select_dtypes(include=['number', 'datetime64'])
                                conv=st.empty()
                                st.subheader(level)
        #                         conv=st.sidebar.radio("Do you want to use this train-test split", ['n', 'y'])
        #                         if conv=='y':
                                ratio=st.sidebar.slider("Enter the train split ratio", 0.20,1.00,0.80, step=0.01)
                                conv1=st.sidebar.radio("Do you want to use this train data ratio", ['n', 'y'])
                                if conv1=='y':
                                    df,df1=split_train_test(df_level,ratio)
                                current_df = df.copy()
                                date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", current_df.select_dtypes(exclude='number').columns.tolist())
                                current_df_date= current_df[[date_ano]].copy()
                                current_df1=df1.copy()
                                conv2=st.sidebar.radio("Do you want to get a overview of data", ['n', 'y'])
                                if conv2=='y':
                                    look_data(current_df, current_df1)
    #                             change_type=st.sidebar.radio("Do you want to change the datatype of any column in the dataset", ['n', 'y'])
    #                             if change_type=='y':
    #                                 current_df, current_df1=change_dtype(current_df, current_df1)
                                pp=st.sidebar.radio("Do you want to proceed with preprocessing", ['n', 'y'])
                                if pp=='y':
                                    history=[]
        #                             i=0
        #                             key="var"
                                    preprocessed_df, preprocessed_df1, history =preprocessing(current_df, current_df1,history)
                                    start_ano= st.button("Want to continue to anomaly detection")
                                    if start_ano:
                                        current_df, current_df1= preprocessed_df, preprocessed_df1
    #                                     date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                                    current_df=current_df.set_index(date_ano)
                                    current_df1=current_df1.set_index(date_ano)
                                    current_df=current_df.select_dtypes(include='number')
                                    current_df1=current_df1.select_dtypes(include='number')
                                    ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                                    ano_fea_cols=[]
                                    for i in ano_fea:
                                        ano_fea_cols.append(i)
                                    current_df=current_df[ano_fea_cols]
                                    current_df1=current_df1[ano_fea_cols]
                                    deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                                    if deseasonalize_res=='y':
                                        current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                                        current_df=current_df.set_index(date_ano)
                                        current_df1=current_df1.set_index(date_ano)
                                    current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                                    X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                                    mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                                    X_mask_model[mask_train]=0
                                    current_df=current_df.set_index(date_ano)
                                    current_df=current_df.apply(lambda x:x.fillna(x.median()))
                                    ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                                    conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                                    if conv2=='y':
                                        current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                                        X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                                    X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                                    st.write(X_train, X_test, X_model_train, X_model_val)
                                    thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                    time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
#                                     batch_size=st.sidebar.slider("Enter the batch size for generating the batches",16,4096,64,step=16)
                                    conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                                    if conv3=='y':
                                        X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                                        X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
#                                         X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv= batch_generator(X_model_train,batch_size), batch_generator(X_model_val,batch_size), batch_generator(X_mask_model_tm,batch_size), batch_generator(X_mask_model_tv,batch_size)
                                        model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                                     thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                        detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)
                                elif pp=='n':
    #                                 date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                                    current_df=current_df.set_index(date_ano)
                                    current_df1=current_df1.set_index(date_ano)
                                    current_df=current_df.select_dtypes(include='number')
                                    current_df1=current_df1.select_dtypes(include='number')
                                    ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                                    ano_fea_cols=[]
                                    for i in ano_fea:
                                        ano_fea_cols.append(i)
                                    current_df=current_df[ano_fea_cols]
                                    current_df1=current_df1[ano_fea_cols]
                                    deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?", ['n','y'])
                                    if deseasonalize_res=='y':
                                        current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                                        current_df=current_df.set_index(date_ano)
                                        current_df1=current_df1.set_index(date_ano)
                                    current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                                    X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                                    mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                                    X_mask_model[mask_train]=0
                                    current_df=current_df.set_index(date_ano)
                                    current_df=current_df.apply(lambda x:x.fillna(x.median()))
                                    ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                                    conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                                    if conv2=='y':
                                        current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                                        X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                                    X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                                    st.write(X_train, X_test, X_model_train, X_model_val)
                                    thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                    time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                                    conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                                    if conv3=='y':
                                        st.write(time_stamps)
                                        X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                                        X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                                        model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                                     thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                        detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)





                elif res=='n':
        #             conv=st.sidebar.radio("Do you want to use this train-test split", ['n', 'y'])
        #             if conv=='y':
                    ratio=st.sidebar.slider("Enter the train split ratio", 0.20,1.00,0.80, step=0.01)
                    conv1=st.sidebar.radio("Do you want to use this train data ratio", ['n', 'y'])
                    if conv1=='y':
                        df,df1=split_train_test(df,ratio)
                    current_df = df.copy()
                    date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", current_df.select_dtypes(exclude='number').columns.tolist())
                    current_df_date= current_df[[date_ano]].copy()
                    current_df1=df1.copy()
                    conv2=st.sidebar.radio("Do you want to get a overview of data", ['n', 'y'])
                    if conv2=='y':
                        look_data(current_df, current_df1)
                    current_df=current_df.select_dtypes(include=['number', 'datetime64'])
                    current_df1=current_df1.select_dtypes(include=['number', 'datetime64'])
                    pp=st.sidebar.selectbox("Do you want to proceed with preprocessing", ['n', 'y'])
                    if (pp=='y'):
                        history=[]
        #                 i=0
        #                 key="var"
                        preprocessed_df, preprocessed_df1, history=preprocessing(current_df, current_df1,history)
                        start_ano= st.button("Want to continue to anomaly detection")
                        if start_ano:
                            current_df, current_df1= preprocessed_df, preprocessed_df1
    #                         date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                        current_df=current_df.set_index(date_ano)
                        current_df1=current_df1.set_index(date_ano)
                        current_df=current_df.select_dtypes(include='number')
                        current_df1=current_df1.select_dtypes(include='number')
                        ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                        ano_fea_cols=[]
                        for i in ano_fea:
                            ano_fea_cols.append(i)
                        current_df=current_df[ano_fea_cols]
                        current_df1=current_df1[ano_fea_cols]
                        deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                        if deseasonalize_res=='y':
                            current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                            current_df=current_df.set_index(date_ano)
                            current_df1=current_df1.set_index(date_ano)
                        current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                        X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                        mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                        X_mask_model[mask_train]=0
                        current_df=current_df.set_index(date_ano)
                        current_df=current_df.apply(lambda x:x.fillna(x.median()))
                        ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                        conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                        if conv2=='y':
                            current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                            X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                        X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                        st.write(X_train, X_test, X_model_train, X_model_val)
                        thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                        time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                        conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                        if conv3=='y':
                            st.write(time_stamps)
                            X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                            X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                            model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                         thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                            detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)
                    elif pp=='n':
    #                     date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                        current_df=current_df.set_index(date_ano)
                        current_df1=current_df1.set_index(date_ano)
                        current_df=current_df.select_dtypes(include='number')
                        current_df1=current_df1.select_dtypes(include='number')
                        ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                        ano_fea_cols=[]
                        for i in ano_fea:
                            ano_fea_cols.append(i)
                        current_df=current_df[ano_fea_cols]
                        current_df1=current_df1[ano_fea_cols]
                        deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                        if deseasonalize_res=='y':
                            current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                            current_df=current_df.set_index(date_ano)
                            current_df1=current_df1.set_index(date_ano)
                        current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                        X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                        mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                        X_mask_model[mask_train]=0
                        current_df=current_df.set_index(date_ano)
                        current_df=current_df.apply(lambda x:x.fillna(x.median()))
                        ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                        conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                        if conv2=='y':
                            current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                            X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                        X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                        st.write(X_train, X_test, X_model_train, X_model_val)
                        thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                        time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                        conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                        if conv3=='y':
                            st.write(time_stamps)
                            X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                            X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                            model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                         thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                            detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)

    elif no_files=="Have train and test data in separate files":
        uploaded_file = st.sidebar.file_uploader("Choose a train dataset file", type=["csv", "xlsx"])
        uploaded_file1 = st.sidebar.file_uploader("Choose a test dataset file", type=["csv", "xlsx"])
        if uploaded_file is not None and uploaded_file1 is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
            df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith("csv") else pd.read_excel(uploaded_file1)
            st.write("Train Dataset Shape:", df.shape)
            st.write(df)
            st.write("Test Dataset Shape:", df1.shape)
            st.write(df1)
            if df is not None and df1 is not None:     
                dtt=st.sidebar.selectbox("Does your data have datetime columns", ['n', 'y'])
                if dtt=='y':
                    df, df1=convert_cols_datetime1(df, df1)
                    
                change_type=st.sidebar.radio("Do you want to change the datatype of any column in the dataset", ['n', 'y'])
                if change_type=='y':
                    df, df1 =change_dtype1(df, df1)
                    
                res=st.sidebar.selectbox("Do you want to take your analysis to a granular level based on categories", ['please select','n', 'y'])
                if res=='y':
                    n = st.sidebar.number_input('How many level types do you have ?',min_value = 1, value = 1)
                    if n>=1: # select the categorical variables only if you have n>=1
                        var = st.sidebar.multiselect('Please select the level columns',df.select_dtypes(exclude=['number','datetime64']).columns.tolist(), max_selections = n)
                        df['key'] = ''
                        df1['key'] = ''
                        level_cols=[]
                        for i in var:
                            level_cols.append(i)
                            df['key'] =  df[i].astype(str) + '-' + df['key']
                            df1['key'] =  df1[i].astype(str) + '-' + df1['key']
                        level =  st.selectbox('Select the category level to preprocess',df['key'].unique().tolist())
                        if level in df['key'].unique().tolist():
                            if df[df['key']==level] is not None and df1[df1['key']==level] is not None:
        #                         df_level=df.select_dtypes(include='number')
                                df_level=df[df['key']==level]
                                df_level1=df1[df1['key']==level]
                                df_level=df_level.select_dtypes(include=['number', 'datetime64'])
                                df_level1=df_level1.select_dtypes(include=['number', 'datetime64'])
                                conv=st.empty()
                                st.subheader(level)
        #                         conv=st.sidebar.radio("Do you want to use this train-test split", ['n', 'y'])
        #                         if conv=='y':
#                                 ratio=st.sidebar.slider("Enter the train split ratio", 0.20,1.00,0.80, step=0.01)
#                                 conv1=st.sidebar.radio("Do you want to use this train data ratio", ['n', 'y'])
#                                 if conv1=='y':
#                                     df,df1=split_train_test(df_level,ratio)
                                current_df = df_level.copy()
                                date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", current_df.select_dtypes(exclude='number').columns.tolist())
                                current_df_date= current_df[[date_ano]].copy()
                                current_df1=df_level1.copy()
                                conv2=st.sidebar.radio("Do you want to get a overview of data", ['n', 'y'])
                                if conv2=='y':
                                    look_data(current_df, current_df1)
    #                             change_type=st.sidebar.radio("Do you want to change the datatype of any column in the dataset", ['n', 'y'])
    #                             if change_type=='y':
    #                                 current_df, current_df1=change_dtype(current_df, current_df1)
                                pp=st.sidebar.radio("Do you want to proceed with preprocessing", ['n', 'y'])
                                if pp=='y':
                                    history=[]
        #                             i=0
        #                             key="var"
                                    preprocessed_df, preprocessed_df1, history =preprocessing(current_df, current_df1,history)
                                    start_ano= st.button("Want to continue to anomaly detection")
                                    if start_ano:
                                        current_df, current_df1= preprocessed_df, preprocessed_df1
    #                                     date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                                    current_df=current_df.set_index(date_ano)
                                    current_df1=current_df1.set_index(date_ano)
                                    current_df=current_df.select_dtypes(include='number')
                                    current_df1=current_df1.select_dtypes(include='number')
                                    ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                                    ano_fea_cols=[]
                                    for i in ano_fea:
                                        ano_fea_cols.append(i)
                                    current_df=current_df[ano_fea_cols]
                                    current_df1=current_df1[ano_fea_cols]
                                    deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                                    if deseasonalize_res=='y':
                                        current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                                        current_df=current_df.set_index(date_ano)
                                        current_df1=current_df1.set_index(date_ano)
                                    current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                                    X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                                    mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                                    X_mask_model[mask_train]=0
                                    current_df=current_df.set_index(date_ano)
                                    current_df=current_df.apply(lambda x:x.fillna(x.median()))
                                    ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                                    conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                                    if conv2=='y':
                                        current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                                        X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                                    X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                                    st.write(X_train, X_test, X_model_train, X_model_val)
                                    thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                    time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                                    conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                                    if conv3=='y':
                                        X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                                        
                                        X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                                        model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                                     thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                        detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)
                                elif pp=='n':
    #                                 date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                                    current_df=current_df.set_index(date_ano)
                                    current_df1=current_df1.set_index(date_ano)
                                    current_df=current_df.select_dtypes(include='number')
                                    current_df1=current_df1.select_dtypes(include='number')
                                    ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                                    ano_fea_cols=[]
                                    for i in ano_fea:
                                        ano_fea_cols.append(i)
                                    current_df=current_df[ano_fea_cols]
                                    current_df1=current_df1[ano_fea_cols]
                                    deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?", ['n','y'])
                                    if deseasonalize_res=='y':
                                        current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                                        current_df=current_df.set_index(date_ano)
                                        current_df1=current_df1.set_index(date_ano)
                                    current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                                    X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                                    mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                                    X_mask_model[mask_train]=0
                                    current_df=current_df.set_index(date_ano)
                                    current_df=current_df.apply(lambda x:x.fillna(x.median()))
                                    ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                                    conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                                    if conv2=='y':
                                        current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                                        X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                                    X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                                    st.write(X_train, X_test, X_model_train, X_model_val)
                                    thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                    time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                                    conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                                    if conv3=='y':
                                        st.write(time_stamps)
                                        X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                                        
                                        X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                                        model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                                     thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                                        detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)





                elif res=='n':
        #             conv=st.sidebar.radio("Do you want to use this train-test split", ['n', 'y'])
        #             if conv=='y':
#                     ratio=st.sidebar.slider("Enter the train split ratio", 0.20,1.00,0.80, step=0.01)
#                     conv1=st.sidebar.radio("Do you want to use this train data ratio", ['n', 'y'])
#                     if conv1=='y':
#                         df,df1=split_train_test(df,ratio)
                    current_df = df.copy()
                    date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", current_df.select_dtypes(exclude='number').columns.tolist())
                    current_df_date= current_df[[date_ano]].copy()
                    current_df1=df1.copy()
                    conv2=st.sidebar.radio("Do you want to get a overview of data", ['n', 'y'])
                    if conv2=='y':
                        look_data(current_df, current_df1)
                    current_df=current_df.select_dtypes(include=['number', 'datetime64'])
                    current_df1=current_df1.select_dtypes(include=['number', 'datetime64'])
                    pp=st.sidebar.selectbox("Do you want to proceed with preprocessing", ['n', 'y'])
                    if (pp=='y'):
                        history=[]
        #                 i=0
        #                 key="var"
                        preprocessed_df, preprocessed_df1, history=preprocessing(current_df, current_df1,history)
                        start_ano= st.button("Want to continue to anomaly detection")
                        if start_ano:
                            current_df, current_df1= preprocessed_df, preprocessed_df1
    #                         date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                        current_df=current_df.set_index(date_ano)
                        current_df1=current_df1.set_index(date_ano)
                        current_df=current_df.select_dtypes(include='number')
                        current_df1=current_df1.select_dtypes(include='number')
                        ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                        ano_fea_cols=[]
                        for i in ano_fea:
                            ano_fea_cols.append(i)
                        current_df=current_df[ano_fea_cols]
                        current_df1=current_df1[ano_fea_cols]
                        deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                        if deseasonalize_res=='y':
                            current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                            current_df=current_df.set_index(date_ano)
                            current_df1=current_df1.set_index(date_ano)
                        current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                        X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                        mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                        X_mask_model[mask_train]=0
                        current_df=current_df.set_index(date_ano)
                        current_df=current_df.apply(lambda x:x.fillna(x.median()))
                        ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                        conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                        if conv2=='y':
                            current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                            X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                        X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                        st.write(X_train, X_test, X_model_train, X_model_val)
                        thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                        time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                        conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                        if conv3=='y':
                            st.write(time_stamps)
                            X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                            
                            X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                            model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                         thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                            detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)
                    elif pp=='n':
    #                     date_ano=st.sidebar.selectbox("Please select the date column for analysis of anomalies", df.select_dtypes(exclude='number').columns.tolist())
                        current_df=current_df.set_index(date_ano)
                        current_df1=current_df1.set_index(date_ano)
                        current_df=current_df.select_dtypes(include='number')
                        current_df1=current_df1.select_dtypes(include='number')
                        ano_fea=st.sidebar.multiselect("Please select the columns to use for anomaly detection", current_df.columns.tolist())
                        ano_fea_cols=[]
                        for i in ano_fea:
                            ano_fea_cols.append(i)
                        current_df=current_df[ano_fea_cols]
                        current_df1=current_df1[ano_fea_cols]
                        deseasonalize_res=st.sidebar.radio("Want to proceed with deseasonalizing the time series data?",['n','y'])
                        if deseasonalize_res=='y':
                            current_df, current_df1=seasonal(current_df, current_df1,date_ano)  
                            current_df=current_df.set_index(date_ano)
                            current_df1=current_df1.set_index(date_ano)
                        current_df=pd.merge(current_df.reset_index(), current_df_date, on=date_ano, how='outer')
                        X_mask_model=np.ones(current_df.drop(date_ano, axis=1).shape)[:,0]
                        mask_train=np.isnan(current_df.drop(date_ano, axis=1).values)[:,0]
                        X_mask_model[mask_train]=0
                        current_df=current_df.set_index(date_ano)
                        current_df=current_df.apply(lambda x:x.fillna(x.median()))
                        ratio_1=st.sidebar.slider("Enter the validation split ratio", 0.10,0.50,0.20, step=0.01)
                        conv2=st.sidebar.radio("Do you want to use this validation data ratio", ['n', 'y'])
                        if conv2=='y':
                            current_df_tm, current_df_tv=split_train_test(current_df, 1-ratio_1)
                            X_mask_model_tm, X_mask_model_tv= X_mask_model[:int(len(X_mask_model)*(1-ratio_1))], X_mask_model[int(len(X_mask_model)*(1-ratio_1)):]
                        X_train, X_test, X_model_train, X_model_val=scale(current_df, current_df1, current_df_tm, current_df_tv)
                        st.write(X_train, X_test, X_model_train, X_model_val)
                        thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                        time_stamps=st.sidebar.slider("Enter the sequence length for generating sequences", 3, 180, 30, step=1)
                        conv3=st.sidebar.radio("Do you want to use this sequence length", ['n', 'y'])
                        if conv3=='y':
                            st.write(time_stamps)
                            X_train, X_test, X_model_train, X_model_val=create_sequences( X_train, X_test, X_model_train, X_model_val, time_stamps)
                            
                            X_mask_model_tm, X_mask_model_tv=create_sequences_mask(X_mask_model_tm, X_mask_model_tv,time_stamps)
                            model=hyperparameter_tuning(X_model_train, X_model_val, X_mask_model_tm, X_mask_model_tv)
    #                         thresh=st.sidebar.slider("Enter the threshold loss percentile for checking the anomalies", 0.0, 100.0, 99.0, step=0.5)
                            detect_anomaly(model,X_train,current_df, X_test, current_df1, current_df.columns.tolist(), time_stamps, '', thresh, 0.05)
        

main()
