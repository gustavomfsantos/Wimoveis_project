# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:40:19 2024

@author: gusta
"""

import os 
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import selenium
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#Paths
base_path = r'C:\Users\gusta\OneDrive\Área de Trabalho\Personal_Projects\Listing_Project'
code_path = base_path + r'\Code'
data_path = base_path + r'\Data'

#Files_names
raw_data = 'data_raw'
transformed_data = 'data_transformed'
final_data = 'data_processed'
#Params
pages = 150

def get_listings( pages):
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
    
    PARAMS
        pages - Number of pages to scrapp
    
    Returns a raw Dataframe with all information collected
    
    """
    
    df_final = pd.DataFrame(columns=['Price', 'Condo_Fee', 'Neighbourhood', 'Size_m2', 
                               'Rooms', 'Bathrooms', 'Garage_Spots', 'Published'])


    total_page = pages
    
    for i in tqdm( range(0,total_page)):


        scrap_link = f'https://www.wimoveis.com.br/apartamentos-venda-brasilia-df-publicado-no-ultimo-mes-pagina-{i}.html'
        
        chrome_options = Options()
        
        chrome_options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome( chrome_options)    
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
        print('Listing Scrapping done!')
        
        df_final = pd.concat([df_final, df])
        df_final.to_csv(data_path + rf'\{raw_data}.csv', index=False)
        print(f'Raw data saved in folder {data_path}')
        
    return




def transform_dataset():
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

    PARAMS
    No parameters

    Returns Data transformed into numerical values and one categorical column
    """
    df_raw = pd.read_csv(data_path + rf'\{raw_data}.csv')    
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
    
    df_processed.to_csv(data_path + rf'\{transformed_data}.csv', index=False)
    print(f'Transformed data saved in folder {data_path}')
    
    return 



def remove_outliers():
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
    
    PARAMS
    No parameters
    
    Returns Final Dataframe with Data Ready for analysis and one step to fit ML models
    
    """

    df = pd.read_csv(data_path + rf'\{transformed_data}.csv')
    print('Initial DF shape', df.shape)
    #desc = (df.describe(include = 'all'))
    #print(desc)

    
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
    
    print('We discarted nearly 100 observations but now we dont have outliers and only have nan values in the Condo Fee column')
    
    #desc1 = (df.describe(include = 'all'))
    #print(desc1)
    print('Dataset Ready for Exploratory Analysis')
    # df['Price_BRL'] = df['Price_BRL'].astype(float)
    df.to_csv(data_path + rf'\{final_data}.csv', index=False)
    print(f'Processed data saved in folder {data_path}')
    
    print('Target Columns Distribution')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    sns.histplot(df['Price_BRL'], bins=10, kde=True, color='skyblue')
    plt.title('Histogram of Continuous Price_BRL')
    plt.xlabel('Target')
    plt.ylabel('Frequency')

    return 


if __name__ == "__main__":

    get_listings( pages)

    transform_dataset()

    remove_outliers()


###############EXPLORATORY ANALYSIS IS IN NOTEBOOK!!!

# df = pd.read_csv(data_path + rf'\{final_data}.csv')
# print('Looking into the 2 listing that don\'t have price tag, they are property release without defined price')
# print('There other columns have some nan values but we will analyze visually.')


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
# sns.boxplot(y=df['Price_BRL'], color='skyblue')
# plt.title('Boxplot of Continuous Target')
# plt.ylabel('Target')


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
# sns.histplot(df['Price_BRL'], bins=10, kde=True, color='skyblue')
# plt.title('Histogram of Continuous Price_BRL')
# plt.xlabel('Target')
# plt.ylabel('Frequency')



# #Relation with neighbours


# #ploting scatter plots to show relation between variables.

# for col in [x for x in df.columns if 'Neighbourhood_Aux'  not in x and  'Price_BRL' not in x]:
#     print(col)

#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(data=df, x=f'{col}', y='Price_BRL')
#     plt.title(f'Scatter Plot of {col} vs Price_BRL')
#     plt.xlabel(f'{col}')
#     plt.ylabel('Target')
    
#     # Show the plot
#     plt.show()
