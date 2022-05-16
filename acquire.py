import pandas as pd
import numpy as np
import os
from env import host, user, password

#Titanic Data

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
def new_titanic_data():
    #This here gives us All from passengers the titanic set
    sql_query = 'SELECT * FROM passengers'
    #return as dataframe(for usability sake)
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    return df

def get_titanic_data():
    #snags a personal copy and gives us a dataframe
    #because what we ain't gonna do is work through 100 arrays
    if os.path.isfile('titanic_df.csv'):
        #to catch if there is already a csv
        #style points
        df = pd.read_csv('titanic_df.csv', index_col=0)
    else:
        #incase not run my old set
        df = new_titanic_data()
        #saves locally
        df.to_csv('titanic_df.csv')
    return df

#Iris database

def new_iris_data():
   # We want just sepal stuff
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    #plug this key into our database
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    return df


def get_iris_data():
    #same as my titanic catcher
    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col=0)
    else:
        df = new_iris_data()
        df.to_csv('iris_df.csv')
    return df

def new_telco_data():
    #Let's pull in telco data, ALL of it.
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    return df

def get_telco_data():
    #like get_iris_data
    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        df = new_telco_data()
        df.to_csv('telco.csv')
    return df