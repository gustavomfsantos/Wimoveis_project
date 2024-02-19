# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:40:19 2024

@author: gusta
"""


import re

import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import uniform, randint
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


from selenium.webdriver.common.by import By
from selenium import webdriver

import pickle
import streamlit as st


def get_listings(data_path, raw_data_name, pages, options):
    """
    This function is to collect listings prices and proprieties information
    on a famous local website. The listing are only for the City Brasilia and
    only apartments, no houses included.
    This functions uses Selenium to go to the website and get all information 
    available for each listing. The website does not support Selenium change pages
    in one unique session, so for each search page selenium will open a new browser
    to collect the information. Because of this work around it might have duplicate
    observations at the end of the process because some listings in a specifi page 
    might appear again in the other page. The search engine in the website does not 
    have a formal and strict order for the results. The duplicates rows will be 
    solved in the next function
    
    Parameters
    ----------
    pages : str
        Number of pages to scrapp
    data_path : str
        raw_data_name: str
    pages: str 
        chrome_options: str
    
    Returns 
    --------
    Raw Dataframe with all information collected
    
    """
    
    df_final = pd.DataFrame(columns=['Price', 'Condo_Fee', 'Neighbourhood', 'Size_m2', 
                               'Rooms', 'Bathrooms', 'Garage_Spots', 'Published'])


    total_page = pages
    
    for i in tqdm( range(1,total_page+1)):


        scrap_link = f'https://www.wimoveis.com.br/apartamentos-venda-brasilia-df-publicado-no-ultimo-mes-pagina-{i}.html'
        

        driver = webdriver.Firefox(options=options)
        driver.get(scrap_link)
                                                      
        div_element  = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]')
        div_text = div_element.text                
        
        split_text = "Contatar"
        
        # Split the big text into individual observations
        observations = div_text.split(split_text)
        
        # Remove leading and trailing whitespace from each observation
        observations = [observation.strip() for observation in observations if observation.strip()]
    
    
        df = pd.DataFrame(columns=['Price', 'Condo_Fee', 'Neighbourhood', 'Size_m2', 
                                   'Rooms', 'Bathrooms', 'Garage_Spots', 'Published', 'Description'])
        
        for observation in observations[1:-1]:
            observation_ = observation.split("\n")
            
            if not [x for x in observation_ if 'R$' in x]:  
                listing_price = np.nan
            else:
                listing_price = [x for x in observation_ if 'R$' in x][0]
            
            if not [x for x in observation_ if 'Condominio' in x and 'R$' in x]:  
                condo_fees = np.nan
            else:
                condo_fees = [x for x in observation_ if 'Condominio' in x and 'R$' in x][0]
            
            if not [x for x in observation_ if 'Brasília' in x or 'Brasilia' in x]:  
                neighbourhood = np.nan
            else:    
                neighbourhood = [x for x in observation_ if 'Brasília' in x or 'Brasilia' in x][0]
            
            if not [x for x in observation_ if 'm²' in x]:  
                size_m2 = np.nan
            else:
                size_m2 = [x for x in observation_ if 'm²' in x][0]
            
            if not [x for x in observation_ if 'quartos' in x]:  
                rooms = np.nan
            else:
                rooms = [x for x in observation_ if 'quartos' in x][0]
            
            if not [x for x in observation_ if 'ban' in x]:  
                bathrooms = np.nan
            else:
                bathrooms = [x for x in observation_ if 'ban' in x][0]
            
            if not [x for x in observation_ if 'vaga' in x]:  
                garage_spots = np.nan
            else:
                garage_spots = [x for x in observation_ if 'vaga' in x][0]
            
            if not [x for x in observation_ if 'Publicado' in x]:  
                published_time = np.nan
            else:
                published_time = [x for x in observation_ if 'Publicado' in x][0]
            
            if not [x for x in observation_ if 'Apartamento' in x or 'Casa' in x]:  
                description = np.nan
            else:
                description = [x for x in observation_ if 'Apartamento' in x or 'Casa' in x][0]
            
            merged_list = list((listing_price, condo_fees, neighbourhood,
                                           size_m2, rooms, bathrooms, garage_spots, published_time, description))
            
            df.loc[len(df)] = merged_list
          
            
        driver.quit()
        print(f'Listing Scrap page {i} done!')
        
        df_final = pd.concat([df_final, df])
        df_final.to_csv(data_path + rf'\{raw_data_name}.csv', index=False)
    print(f'Raw data saved in folder {data_path}')
        
    return


def transform_dataset(data_path, raw_data_name, transformed_data_name):
    """
    This function transform the scrapped raw data into a preliminar usable dataset
    The scrap process collects string information and, since it is a scrapp on
    a website that does not have a pattern to display the information needed,
    we have to check if the info in each column makes sense for the column. 
    Listing sites are only a plataform for users list they properties. Each user
    has different way to write and display information. This function will convert
    string values into numeric and also will try to look for missing information
    in the listing description in order to get data that the scrapp was not able to get .
    At the end, this function will drop duplicates, convert columns and replace 
    NAN with certain conditions    
    
    Parameters
    ----------
    data_path : str
        Path to save and load dataset
    raw_data_name : str
        Name given to raw dataset
    transformed_data_name : str
        Name given to transformed dataset
        
    Returns 
    -------
    Data transformed into numerical values and one categorical column
    """
    df_raw = pd.read_csv(data_path + rf'\{raw_data_name}.csv')    
    df_raw = df_raw.drop_duplicates()
    print('Initial DF shape', df_raw.shape)
    #First column to ajust is the listing price, to get only the value in BRL currency
    df_raw['Price_BRL'] = df_raw['Price'].str[3:]
    df_raw['Price_BRL'] =  df_raw['Price_BRL'].str.replace('.', '').astype(float)
    
    #Condo fee Feature
    df_raw['Condo_Fee_BRL'] = df_raw['Condo_Fee'].str[3:-10]
    df_raw.loc[df_raw['Condo_Fee_BRL'].str.len() > 15, 'Condo_Fee_BRL'] = np.nan
    df_raw['Condo_Fee_BRL'] =  df_raw['Condo_Fee_BRL'].str.replace('.', '').astype(float)
    
    #Neighbourhood
    df_raw['Neighbourhood_Aux'] = df_raw['Neighbourhood'].str.split(',').str[0]
    
    #Area
    df_raw['Apt_Area_m2'] = df_raw['Size_m2'].str[:-8]
    df_raw.loc[df_raw['Apt_Area_m2'].str.len() > 6, 'Apt_Area_m2'] = np.nan
    df_raw['Apt_Area_m2'] =  df_raw['Apt_Area_m2'].str.replace('.', '').astype(float)
    
    #Rooms
    df_raw['Total_Rooms'] = df_raw['Rooms'].str[:-8]
    df_raw.loc[df_raw['Total_Rooms'].str.len() > 2, 'Total_Rooms'] = np.nan
    df_raw['Total_Rooms'] =  df_raw['Total_Rooms'].str.replace('.', '').astype(float)
    
    
    #Baths
    df_raw['Total_Baths'] = df_raw['Bathrooms']
    df_raw.loc[df_raw['Total_Baths'].str.len() > 15, 'Total_Baths'] = np.nan
    df_raw['Total_Baths'] = df_raw['Total_Baths'].str[:1]
    df_raw['Total_Baths'] =  df_raw['Total_Baths'].str.replace('.', '').astype(float)
    
    
    #Garage Spots
    df_raw['Garage_Spots_aux'] = df_raw['Garage_Spots'].str[:-4]
    df_raw.loc[df_raw['Garage_Spots_aux'].str.len() > 2, 'Garage_Spots_aux'] = np.nan
    df_raw['Garage_Spots_aux'] =  df_raw['Garage_Spots_aux'].str.replace('.', '').astype(float)
    
    #Days Published
    df_raw['Days_Published'] = df_raw['Published'].str[-7:-4]
    df_raw.loc[df_raw['Days_Published'].str.len() > 4, 'Days_Published'] = np.nan
    rows_to_replace = df_raw['Days_Published'].str.contains('o').fillna(False) #For listing published today or yesterday, the format is different. Assume value 1 for those
    
    df_raw.loc[rows_to_replace, 'Days_Published'] = 1 
    df_raw['Days_Published'] =  df_raw['Days_Published'].str.replace('.', '').astype(float)
    
    
    #Get only columns needed
    df_process = df_raw.iloc[:, 8:]
    
    
    
    #Garage
    garage_keywords = ['vaga','vagas']
    garage_pattern = r'(\w+)\s+\b(?:{})\b'.format('|'.join(garage_keywords))
    # Extract strings next to keywords
    df_process['Descr_garage'] = df_process['Description'].str.extract(garage_pattern, flags=re.IGNORECASE)
    df_process.loc[df_process['Descr_garage'].str.len() > 2, 'Descr_garage'] = np.nan
    try:
        df_process['Descr_garage'] = df_process['Descr_garage'].astype(float)
    except ValueError:
        df_process['Descr_garage']= pd.to_numeric(df_process['Descr_garage'], errors='coerce')
    
    nan_count_before = df_process['Garage_Spots_aux'].isna().sum()
    df_process['Garage_Spots_aux'] = df_process['Garage_Spots_aux'].fillna(df_process['Descr_garage'])
    nan_count_after = df_process['Garage_Spots_aux'].isna().sum()
    nan_replaced = nan_count_before - nan_count_after
    print("Number of NaN values replaced for garage spots:", nan_replaced)
    
    
    #Room
    room_keywords = ['quarto','quartos']
    room_pattern = r'(\w+)\s+\b(?:{})\b'.format('|'.join(room_keywords))
    # Extract strings next to keywords
    df_process['Descr_room'] = df_process['Description'].str.extract(room_pattern, flags=re.IGNORECASE)
    df_process.loc[df_process['Descr_room'].str.len() > 2, 'Descr_room'] = np.nan
    try:
        df_process['Descr_room'] = df_process['Descr_room'].astype(float)
    except ValueError:
        df_process['Descr_room']= pd.to_numeric(df_process['Descr_room'], errors='coerce')
    
    nan_count_before = df_process['Total_Rooms'].isna().sum()
    df_process['Total_Rooms'] = df_process['Total_Rooms'].fillna(df_process['Descr_room'])
    nan_count_after = df_process['Total_Rooms'].isna().sum()
    nan_replaced = nan_count_before - nan_count_after
    print("Number of NaN values replaced for Rooms:", nan_replaced)
    
    
    #Bathrooms
    bath_keywords = ['banheiro','banheiros']
    bath_pattern = r'(\w+)\s+\b(?:{})\b'.format('|'.join(bath_keywords))
    # Extract strings next to keywords
    df_process['Descr_bath'] = df_process['Description'].str.extract(bath_pattern, flags=re.IGNORECASE)
    df_process.loc[df_process['Descr_bath'].str.len() > 2, 'Descr_bath'] = np.nan
    try:
        df_process['Descr_bath'] = df_process['Descr_bath'].astype(float)
    except ValueError:
        df_process['Descr_bath']= pd.to_numeric(df_process['Descr_bath'], errors='coerce')
        
    nan_count_before = df_process['Total_Baths'].isna().sum()
    df_process['Total_Baths'] = df_process['Total_Baths'].fillna(df_process['Descr_bath'])
    nan_count_after = df_process['Total_Baths'].isna().sum()
    nan_replaced = nan_count_before - nan_count_after
    print("Number of NaN values replaced for Bathrooms:", nan_replaced)
        
    
    #Condo fee
    condo_keywords = [ 'Condomínio no valor de', 'Condomínio']
    condo_pattern =  r'\b(?:{})\b(?:(?!\b(?:{})\b).)*?\s+(\w+)'.format('|'.join(condo_keywords), '|'.join(condo_keywords[1:]))
    # Extract strings next to keywords
    df_process['Descr_condo'] = df_process['Description'].str.extract(condo_pattern, flags=re.IGNORECASE)#.apply(lambda row: ' '.join(row.dropna()), axis=1)
    df_process.loc[df_process['Descr_condo'].str.len() < 2, 'Descr_condo'] = np.nan
    try:
        df_process['Descr_condo'] = df_process['Descr_condo'].astype(float)
    except ValueError:
        df_process['Descr_condo']= pd.to_numeric(df_process['Descr_condo'], errors='coerce')    
    
    nan_count_before = df_process['Condo_Fee_BRL'].isna().sum()
    df_process['Condo_Fee_BRL'] = df_process['Condo_Fee_BRL'].fillna(df_process['Descr_condo'])
    nan_count_after = df_process['Condo_Fee_BRL'].isna().sum()
    nan_replaced = nan_count_before - nan_count_after
    print("Number of NaN values replaced for Condo Fees:", nan_replaced)
    
    
    #Area size
    area_keywords = ['metros','de area', 'm²']
    area_pattern = r'(\w+)\s+\b(?:{})\b'.format('|'.join(area_keywords))
    # Extract strings next to keywords
    df_process['Descr_area'] = df_process['Description'].str.extract(area_pattern, flags=re.IGNORECASE)
    df_process.loc[df_process['Descr_area'].str.len() > 3, 'Descr_area'] = np.nan
    try:
        df_process['Descr_area'] = df_process['Descr_area'].astype(float)
    except ValueError:
        df_process['Descr_area']= pd.to_numeric(df_process['Descr_area'], errors='coerce')
    
    nan_count_before = df_process['Apt_Area_m2'].isna().sum()
    df_process['Apt_Area_m2'] = df_process['Apt_Area_m2'].fillna(df_process['Descr_area'])
    nan_count_after = df_process['Apt_Area_m2'].isna().sum()
    nan_replaced = nan_count_before - nan_count_after
    print("Number of NaN values replaced for Area:", nan_replaced)
    
    print('Final DF shape', df_raw.shape)
    df_processed = df_process.drop([x for x in df_process.columns if 'Descr' in x], axis = 1)
    print('Dataset in correct format')
    
    df_processed.to_csv(data_path + rf'\{transformed_data_name}.csv', index=False)
    print(f'Transformed data saved in folder {data_path}')
    
    return 


def remove_outliers(data_path, transformed_data_name, final_data_name):
    """
    This function remove outliers in all features.
    It also deal with NAN values, discarting rows with Null values in columns, 
    execpt the Condo_Fee_BRL, Garage_Spots_aux and Days_Published columns.
    For rows with Condo Fee NAN no action was take. For other two it was replaced
    by 0 for Garage Spot, assuming lack of information is the lack of garage spots
    and for days published it is assumed that those rows are sponsor listing and 
    we assumed value 1 for those to treat as a fresh listing.
    The bonds for the variables are:
        Price_BRL          200,000 - 5,000,000
        Condo_Fee_BRL      100 - 5,000
        Neighbourhood_Aux  String Column, no action
        Apt_Area_m2        20m - 600m
        Total_Rooms        1 - 7
        Total_Baths        1 - 7
        Garage_Spots_aux   0 - 6
        Days_Published     1 - 30
    
    Target Columns Distribution indicates that that are still outliers but
    the proprieties market has luxury apartaments that will be very expensive
    and eliminating them from our analysis will cause information lose.
    And also will get specific address in neighbour column and transform
    into the correct neighbourhood related to the address
    
    Parameters
    ----------
    data_path : str
        Path to save and load dataset
    transformed_data_name : str
        Name given to transformed dataset
    final_data_name : str
            Name given to final dataset
    
    Returns
    -------
    Final Dataframe with Data Ready for analysis and one step to fit ML models
    
    """
    
    df = pd.read_csv(data_path + rf'\{transformed_data_name}.csv')
    print('Initial DF shape', df.shape)
    
    
    df = df[(df['Price_BRL'] >= 200000) & (df['Price_BRL'] <= 5000000)]
    
    df = df[((df['Condo_Fee_BRL'].isnull()) | ((df['Condo_Fee_BRL'] >= 100) & (df['Condo_Fee_BRL'] <= 5000)))]
    
    df = df[ ((df['Apt_Area_m2'] >= 20) & (df['Apt_Area_m2'] <= 600))]
    
    
    df = df[((df['Total_Rooms'].notnull()) | ((df['Total_Rooms'] >= 1) & (df['Total_Rooms'] <= 7)))]
    df = df[((df['Total_Baths'].notnull()) | ((df['Total_Baths'] >= 1) & (df['Total_Baths'] <= 7)))]
        
    
    df['Garage_Spots_aux'] = df['Garage_Spots_aux'].fillna(0)
    df = df[(df['Garage_Spots_aux'] >= 0) & (df['Garage_Spots_aux'] <= 6)]
    
        
    df['Days_Published'] = df['Days_Published'].fillna(1)
    df = df[(df['Days_Published'] >= 1) & (df['Days_Published'] <= 30)]
    
    
    print('Final DF shape', df.shape)
    print(df.info())
    
    
    noroeste_to_search = 'sqnw'
    noroeste_string = 'Noroeste'
    mask_noroeste = df['Neighbourhood_Aux'].str.contains(noroeste_to_search, case=False)
    df.loc[mask_noroeste, 'Neighbourhood_Aux'] = noroeste_string
    
    
    noroeste_to_search = 'Noroeste'
    noroeste_string = 'Noroeste'
    mask_noroeste = df['Neighbourhood_Aux'].str.contains(noroeste_to_search, case=False)
    df.loc[mask_noroeste, 'Neighbourhood_Aux'] = noroeste_string
    
    
    noroeste_to_search = 'SHCNW'
    noroeste_string = 'Noroeste'
    mask_noroeste = df['Neighbourhood_Aux'].str.contains(noroeste_to_search, case=False)
    df.loc[mask_noroeste, 'Neighbourhood_Aux'] = noroeste_string
    
    
    sudoeste_to_search = 'Sudoeste'
    sudoeste_string = 'Sudoeste'
    mask_sudoeste = df['Neighbourhood_Aux'].str.contains(sudoeste_to_search, case=False)
    df.loc[mask_sudoeste, 'Neighbourhood_Aux'] = sudoeste_string
        
    
    sudoeste_to_search = 'SQSW'
    sudoeste_string = 'Sudoeste'
    mask_sudoeste = df['Neighbourhood_Aux'].str.contains(sudoeste_to_search, case=False)
    df.loc[mask_sudoeste, 'Neighbourhood_Aux'] = sudoeste_string
    
    
    asanorte_to_search = 'CLN'
    asanorte_string = 'Asa Norte'
    mask_asanorte = df['Neighbourhood_Aux'].str.contains(asanorte_to_search, case=False)
    df.loc[mask_asanorte, 'Neighbourhood_Aux'] = asanorte_string
    
    
    asanorte_to_search = 'SQN'
    asanorte_string = 'Asa Norte'
    mask_asanorte = df['Neighbourhood_Aux'].str.contains(asanorte_to_search, case=False)
    df.loc[mask_asanorte, 'Neighbourhood_Aux'] = asanorte_string
    
    
    lagonorte_to_search = 'SHI'
    lagonorte_string = 'Lago Norte'
    mask_lagonorte = df['Neighbourhood_Aux'].str.contains(lagonorte_to_search, case=False)
    df.loc[mask_lagonorte, 'Neighbourhood_Aux'] = lagonorte_string
    
    
    asasul_to_search = 'Quadra Sul'
    asasul_string = 'Asa Sul'
    mask_asasul = df['Neighbourhood_Aux'].str.contains(asasul_to_search, case=False)
    df.loc[mask_asasul, 'Neighbourhood_Aux'] = asasul_string
    
    
    parksul_to_search = 'vista park'
    parksul_string = 'Park Sul'
    mask_parksul = df['Neighbourhood_Aux'].str.contains(parksul_to_search, case=False)
    df.loc[mask_parksul, 'Neighbourhood_Aux'] = parksul_string
    
    
    
    #Remove address that have less than one appeareance
    category_counts = df['Neighbourhood_Aux'].value_counts()
    
    # Filter out categories with counts less than one
    valid_categories = category_counts[category_counts > 1].index
    
    # Filter the DataFrame to keep only rows with valid categories
    df = df[df['Neighbourhood_Aux'].isin(valid_categories)]
    
    # Drop Brasilia as has no information regarding neighbourhood
    df = df[df['Neighbourhood_Aux'] != 'Brasília']
    
    print('We discarted nearly 100 observations but now we dont have outliers and only have nan values in the Condo Fee column')
    
    #desc1 = (df.describe(include = 'all'))
    #print(desc1)
    print('Dataset Ready for Exploratory Analysis')
    # df['Price_BRL'] = df['Price_BRL'].astype(float)
    df.to_csv(data_path + rf'\{final_data_name}.csv', index=False)
    print(f'Processed data saved in folder {data_path}')
    
    print('Target Columns Distribution')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    sns.histplot(df['Price_BRL'], bins=10, kde=True, color='skyblue')
    plt.title('Histogram of Price_BRL Million')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    
    return 


def final_adjust(data_path, final_data_name, ml_data):
    """
    

    Parameters
    ----------
    data_path : str
        Folder path to data files
    final_data_name : str
        Final data file name. Final data is used for analysis but not suited to fit the model
    ml_data : str
        ML data file name. ML data is the data to fit the model

    Returns
    -------
    None.

    """
    
    
    df = pd.read_csv(data_path + rf'\{final_data_name}.csv')
    print('Initial DF shape', df.shape)
    
    
    df_binary = pd.get_dummies(df['Neighbourhood_Aux'], drop_first=True)
    df_binary = df_binary.astype(int)
    df.drop(['Neighbourhood_Aux', 'Condo_Fee_BRL', 'Days_Published'], axis=1, inplace=True)
    
    # Concatenate the original DataFrame with the binary columns
    df_final = pd.concat([df, df_binary], axis=1)
    
    #The intercept will assume value for asa norte neighbourhood
    #The intercept term in the logistic regression model captures the baseline probability when all binary dummy variables are 0
    
    #Drop condo fee column, to many null
    print('Final DF shape', df_final.shape)
    
    df_final.to_csv(data_path + rf'\{ml_data}.csv', index=False)
    
    return


def parameters_grid(model_list):
    """
    Alpha determines the relative weight of the ridge and lasso penalties, while lambda determines the overall strength of the regularization
    
    Parameters
    ----------
    model_list : list of str
        List contain models to be trained
        
    Returns
    -------
    paramater_dict : Dict
        Dictionary with hyperparameters to use when tunning the model
        
    """
        
        
    paramater_dict = {}
    for model in model_list:
        if model == 'Lasso':
            param_grid = {
            'alpha': uniform(0.1, 1.0),  # Range for alpha
            'fit_intercept': [True]  # Intercept must be in because of categorical binary transformation left one category out, so intercept will assume this value
            }
            
            paramater_dict[model] = param_grid
            
        if model == 'EN':
            param_grid = {
            'alpha': uniform(0.1, 1.0),  # Range for alpha
            'l1_ratio': uniform(0.0, 1.0),  # Range for l1_ratio
            'fit_intercept': [True]  # Values for fit_intercept
            }
            
            paramater_dict[model] = param_grid   
            
            
    return paramater_dict


def model_inference(data_path, ml_data, random_numb, folds, tune_tryouts, 
                    target, model_list, model_params_grid, model_path):    
    """
    R2 does not measure if the right model was chosen or the predictive capacity of trained model.
    R-squared is meaningful only in the case where the estimated linear regression model is statistically adequate
    Which we can assume this case. The goal of this model is to get the parameters/estimators for each feature used
    and use it to calculated a expected listing price given specific characteristics;
    Is not used that often and some even dont recommend it to use, but for this case is a good fit.
    
    R-squared is a goodness-of-fit measure for linear regression models. 
    This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. 
    R-squared measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale.
    
    We care more about accurately estimating the co-efficient of the Independent variables as accurate estimates help to make 
    inferences and establish causality. we want the model to be simpler and easy to interpret
    Parameters
    ----------
    data_path : str
        Folder path to the data folder
    ml_data : str
        File name of data ready to fit the model.
    random_numb : TYPE
        Random number to replicated the process.
    folds : int
        Number os cross validation folds/partitions.
    tune_tryouts : int
        Number of iterations done when tunning the model.
    target : str
        Target feature - dependent variable
    model_list : list of str
        List containing the models to be trained.
    model_params_grid : dict
        Dictionary with hyperparameters to use on models
    model_path : str
        Folder path to the model and coefficient folder
    Returns
    -------
    Create final model pickle file and coefficients csv file to use on the app deployment
        
    """
    df = pd.read_csv(data_path + rf'\{ml_data}.csv')
    print('Initial DF shape', df.shape)
    
    #Split dataset randomly, eventhough there will be imbalanced neighbourhood columns
    #That is something that I have to accept it.
    #There is no time relation between observations so no special split necessary
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis = 1), df[target], test_size=0.2, random_state=random_numb)
    
    
    #Elastic Net better use for this task!   
    
    
    enet = ElasticNet()
    for model in model_list:
        
        # Create RandomizedSearchCV object
        random_search = RandomizedSearchCV(estimator=enet, 
            param_distributions=model_params_grid['EN'], n_iter=tune_tryouts, 
            cv=folds, random_state=random_numb, scoring = 'r2')
        
        # Fit the model
        random_search.fit(X_train, y_train)
        
        # Get best parameters and best score
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
    
        # 1. Test the model on the test dataset
        best_estimator = random_search.best_estimator_
            
        # 1. Train the model using the best estimators
        best_estimator.fit(X_train, y_train)
        
        # 2. Test the model on the test dataset
        y_pred = pd.Series( best_estimator.predict(X_test))
        accuracy = r2_score(y_test, y_pred)
        print("Accuracy on test set:", accuracy)
       
    #Save only one model for now
    with open(model_path + '/trained.pkl', 'wb') as file:
        pickle.dump(best_estimator, file)
        
        
    columns = X_train.columns
    
    coefficients = best_estimator.coef_
    intercept = best_estimator.intercept_
    coef_column_mapping = dict(zip(columns, coefficients))
    coef_column_mapping['Intercept'] = intercept
    
    # Convert the dictionary to a DataFrame for easier inspection
    coef_df = pd.DataFrame.from_dict(coef_column_mapping, orient='index', columns=['Value'])
    # Save coefficients to a CSV file
    coef_df.to_csv(model_path + '/coefficients.csv')
        
        #print(coef_df)
    
    return 'Model done'
    
    

if __name__ == "__main__":

    #Paths
    base_path = r'C:\Users\gusta\OneDrive\Área de Trabalho\Personal_Projects\Listing_Project'
    code_path = base_path + '/Code'
    data_path = base_path + '/Data'
    model_path = base_path + '/Model'
    #Files_names
    raw_data_name = 'data_raw'
    transformed_data_name = 'data_transformed'
    final_data_name = 'data_processed'
    ml_data = 'data_ml'
    #Params
    pages = 150
    
    
    options = webdriver.FirefoxOptions()
    options.add_argument("--window-size=1920,1080")
    #options.add_argument("--headless")
    
    # get_listings(data_path, raw_data_name, pages, options)
    
    # transform_dataset(data_path, raw_data_name, transformed_data_name)

    # remove_outliers(data_path, transformed_data_name, final_data_name)

    model_list = ['EN']
    target = 'Price_BRL'
    random_numb = 42
    folds = 5
    tune_tryouts = 100
    
    final_adjust(data_path, final_data_name, ml_data)
    model_params_grid = parameters_grid(model_list)
    model_inference(data_path, ml_data, random_numb, folds, tune_tryouts, 
                        target, model_list, model_params_grid, model_path)