
#Brasilia Apartment Price Estimation Project
##Overview
This project aims to estimate fair prices for apartments in Brasilia using market data collected from the local listing website Wimoveis. The process involves scraping data from the website, performing data engineering and feature engineering, analyzing the data using Databricks notebooks, and fitting an elastic net model to estimate apartment prices based on various features.

###Project Structure
Data Scraping: The initial step involves using Selenium to scrape data from the Wimoveis website. All data is collected as text for further processing.

###Data Engineering: This step involves cleaning and structuring the scraped data to make it suitable for analysis. It may include tasks such as handling missing values, formatting data types, and removing duplicates.

###Feature Engineering: 
Features are created or transformed to improve the performance of the predictive model. This step may involve feature scaling, encoding categorical variables, or creating new features based on domain knowledge.

###Data Analysis: 
Data analysis is performed using Databricks notebooks. This includes exploratory data analysis (EDA) to understand the relationships between variables and visualize insights from the data.

###Modeling: 
An elastic net regression model is fit to the data to estimate apartment prices. This model considers the effects of various features on the price and helps in understanding their importance.

###Inference Tool: 
The coefficients obtained from the elastic net model are used to build an inference tool. This tool allows users to estimate the price of an apartment based on selected features such as the number of rooms, apartment area, bathrooms, and neighborhood.

###Dependencies
Selenium
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

###Usage
Clone this repository to your local machine.
Install the necessary dependencies using pip install -r requirements.txt.
Run the Jupyter notebooks in the specified order to execute each step of the project.
Use the inference tool to estimate apartment prices based on selected features.