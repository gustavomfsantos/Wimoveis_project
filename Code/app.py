# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:01:29 2024

@author: gusta
"""
import streamlit as st
import pandas as pd



def stramlit_app(model_path):
    """
    

    Parameters
    ----------
    model_path : str
        Folder path to the model and coefficient folder

    Returns
    -------
    Local application to make estimation regarding apartment prices in Brasilia
    using the trained model coeficients

    """
    
    coef_df = pd.read_csv(model_path + '/coefficients.csv', index_col=0)

    # Set default value for intercept
    transformed_series, unique_value_intercept = pd.factorize(coef_df.loc['Intercept'], sort=True)
    default_intercept = unique_value_intercept
    
    
    # Create a Streamlit dashboard
    st.title('Brasilia Apartment Price Estimation')
    
    st.write("Default Intercept Value is related to Asa Norte Apartment with no other features. Need to input all of it!")
    
    neighbourhood_categories = ['Asa Sul', 'Cruzeiro', 'Guará',
    'Guará II', 'Lago Norte', 'Lago Sul', 'Noroeste', 'Octogonal',
    'Park Sul', 'Samambaia', 'Setor De Clubes Esportivos Sul',
    'Setor Habitacional Jardins Mangueiral',
    'Setor de Hotéis e Turismo Norte', 'Sobradinho', 'Sudoeste',
    'Taguatinga', 'Águas Claras'] #Asa_norte is the intercept
    
    # Allow user to select a category for binary columns
    categorical_columns = neighbourhood_categories
    category_options = ['Asa Norte'] + categorical_columns
    selected_category = st.selectbox('Select a category', category_options)
    
    numeric_columns = [col for col in coef_df.index if col not in categorical_columns and 'Intercept' not in col]

    # Allow user to input feature values
    feature_values = {}
    for column in numeric_columns:
        feature_values[column] = st.number_input(f'Enter value for {column}', step=0.1)
    
    # Calculate predicted target value
    predicted_target = default_intercept
    for column in numeric_columns:
        predicted_target += feature_values[column] * coef_df.loc[column, 'Value']
    
    # If a specific category is selected, add the coefficient for that category
    if selected_category != 'Asa Norte':
        predicted_target += coef_df.loc[selected_category, 'Value']

    predicted_target = round(predicted_target[0], 0)
    predicted_target = int(predicted_target)
    # Display predicted target value
    st.write(f'Apartment Expected Price According to Market: {predicted_target:,} BRL')
    
    return

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

    
    stramlit_app(model_path)