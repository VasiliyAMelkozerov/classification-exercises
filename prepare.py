import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Iris data
def clean_iris(df):
    #Drop the species_id and measurement_id columns.
    df = df.drop(columns=['species_id','measurement_id'])
    #Rename the species_name column to just species.
    df = df.rename(columns={'species_name':'species'})
    #Create dummy variables of the species name and concatenate onto the iris dataframe
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    #concatenate onto the iris dataframe
    df = pd.concat([df, dummy_df], axis=1)
    return df

def cleen_iris(df):
    #Drop the species_id and measurement_id columns.
    df = df.drop(columns =['species_id','measurement_id'])
    #Rename the species_name column to just species.
    df = df.rename(columns = {'species_name':'species'})
    #Create dummy variables of the species name and concatenate onto the iris dataframe. 
    #(This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
    dummy_df = pd.get_dummies(df['species'], drop_first = False)
    df = df.concat([df, dummy_df], axis = 1)

def split_iris_data(df):
    #recieve data frame return values train validate and test
    # splits df to train_validate and test species here as stratify to get balanced view
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    #where the magic happens again making sure to stratify species
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.species)
    return train, validate, test

def prep_iris(df):
    #Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.
    df = df.drop(columns='species_id')
    df = df.rename(columns={'species_name':'species'})
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)
    train, validate, test = split_iris_data(df)
    return train, validate, test

#titanic data

def clean_titanic_data(df):
    #Drop any unnecessary, unhelpful, or duplicated columns.
    df = df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class']
    df = df.drop(columns=cols_to_drop)
    #fill in some empty emark_town's to fit in with context
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    #Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])
    #turn sex and embark_town into code friendly binary flags
    df = pd.concat([df, dummy_df], axis=1)
    return df

def split_titanic_data(df):
    #setup our variables for the follwoing function
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    #gives us our train validate and test variables
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    return train, validate, test

def impute_titanic_mode(train, validate, test):
    #impute is to fill in missing values
    #we let the computer do the work to find the best fit
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def impute_mean_age(train, validate, test):
    #this gives us train validate test variables for age when not given
    #find what value we want to fill with
    imputer = SimpleImputer(strategy = 'mean')
    #setup to get train var
    train['age'] = imputer.fit_transform(train[['age']])
    #setup validate var
    validate['age'] = imputer.transform(validate[['age']])    
    #setup age into test
    test['age'] = imputer.transform(test[['age']])
    return train, validate, test

#Create a function named prep_titanic that accepts the raw titanic data, and returns the data with the transformations above applied.

def prep_titanic_data(df):
    #takes in our modified titanic and gives us train validate and test variables
    df = clean_titanic_data(df)
    train, validate, test = split_titanic_data(df)
    train, validate, test = impute_mean_age(train, validate, test)
    return train, validate, test

#Telco data

def split_telco_data(df):
    #we want to see train validate and test from a more usuable churn reading
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.churn)
    return train, validate, test

#Create a function named prep_telco that accepts the raw telco data, and returns the data with the transformations above applied.

def prep_telco_data(df):
    #Drop duplicate columns
    #Drop any unnecessary, unhelpful, or duplicated columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    #REmove whitspace values  
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    #change to float
    df['total_charges'] = df.total_charges.astype(float)
    #turn the following into computer friendly binaries
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    #Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
    dummy_df = pd.get_dummies(df[['multiple_lines', \
        'online_security', \
        'online_backup', \
        'device_protection', \
        'tech_support', \
        'streaming_tv', \
        'streaming_movies', \
        'contract_type', \
        'internet_service_type', \
        'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    # split the data
    train, validate, test = split_telco_data(df)
    return train, validate, test