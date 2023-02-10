# Loading in required libraries
import pandas as pd
import numpy as np

def create_time_bound_claims(medical_claims, input_code):
    # Removes rows of the medical claims dataframe where the Header Service From Date is equal to or greater than the first Header Service From Date where the input base ICD10 code occurred.
    # Inputs:
    # medical_claims: The medical claims dataframe that will be used to create the time bound dataframe
    # input_code: The ICD10 base code used to create the time bound dataframe  
    
    # Create a min_date_df dataframe that will be joined into the medical_claims dataframe
    min_date_df = medical_claims[['Member Life ID', 'Header Service From Date', 'Base Code']]
    min_date_df = min_date_df[min_date_df['Base Code'] == input_code] 
    min_date_df = min_date_df[['Member Life ID', 'Header Service From Date']]
    min_data_df = min_date_df.groupby('Member Life ID').min().reset_index()
    min_data_df = min_data_df.rename(columns = {'Header Service From Date': 'min_date'})

    # Merging the medical_claims and min_date_df into an augmented_medical_claims dataframe
    augmented_medical_claims = pd.merge(medical_claims, min_data_df, how = 'left', on = 'Member Life ID')

    # Selecting rows
    filtered_medical_claims = augmented_medical_claims[(augmented_medical_claims['Header Service From Date'] < augmented_medical_claims['min_date']) | augmented_medical_claims['min_date'].isna() |\
     ((augmented_medical_claims['Base Code'] == input_code) & (augmented_medical_claims['Header Service From Date'] == augmented_medical_claims['min_date']))].reset_index(drop=True) 

    return filtered_medical_claims

def gen_one_hot_data_input(medical_claims, rx_claims):
    # Takes the input medical_claims and rx_claims dataframes and outputs a dataset with one-hot encoded base claims columns and Biological Gende, Line Service From Date, Header Service From Date, and Birth Date columns.
    # Differs from the function below in that this function requires that the user input a list of base codes they want added to the one-hot encoded dataframe.
    # Inputs: 
    # medical_claims = a dataframe containing the medical claims data where the icd10 base codes have been melted into one column (datasets used include one with all the icd10 codes and one with only primary codes.)
    # rx_claims = a dataframe containing the rx claims data
    # Get the unique Patient Life IDs from the medical claims data
    unique_patient_ids = medical_claims['Member Life ID'].unique()

    # Create a new dataframe to hold input data
    input_data = pd.DataFrame({'Member Life ID':unique_patient_ids})

    base_codes = np.unique(medical_claims['Base Code'])

    # Adding one-hot encoded indicators of possibly related diagnoses
    for base_code in base_codes:
        input_data[base_code] = 0

    for base_code in base_codes:
        current_slice = medical_claims.loc[medical_claims['Base Code'] == base_code]

        current_diag_ids = current_slice['Member Life ID'].unique()

        input_data[base_code].loc[input_data['Member Life ID'].isin(current_diag_ids)] = 1

    # Left join in the gender data from the medical claims dataframe
    gender_data = medical_claims[['Member Life ID', 'Biological Gender']].drop_duplicates()
    input_data = input_data.merge(gender_data, how = 'left', left_on = 'Member Life ID', right_on = 'Member Life ID')
    input_data = input_data.drop_duplicates()

    input_data = input_data.reset_index()
    input_data = input_data.drop('index', axis = 1)

    # Get the earliest Line Service From Date for each Member Life ID
    line_service_data = medical_claims[['Member Life ID', 'Line Service From Date']]

    line_service_data = line_service_data.groupby(['Member Life ID']).min()

    # Get the earliest Header Service From Date
    header_service_data = medical_claims[['Member Life ID', 'Header Service From Date']]

    header_service_data = header_service_data.groupby(['Member Life ID']).min()

    # Left join in the Line Service From Date medical claims dataframe
    input_data = input_data.merge(line_service_data, how = 'left', left_on = 'Member Life ID', right_on = 'Member Life ID')
    input_data = input_data.drop_duplicates()

    input_data = input_data.reset_index()
    input_data = input_data.drop('index', axis = 1)

    # Left join in the Headaer Service From Date data from the medical claims dataframe
    input_data = input_data.merge(header_service_data, how = 'left', left_on = 'Member Life ID', right_on = 'Member Life ID')
    input_data = input_data.drop_duplicates()

    input_data = input_data.reset_index()
    input_data = input_data.drop('index', axis = 1)

    # Left join in the Date of Birth data from the pharmacy dataframe
    rx_birthdates = rx_claims[['Member Life ID', 'Birth Date']]
    rx_birthdates = rx_birthdates.drop_duplicates()
    rx_birthdates

    # Left join in the gender data from the medical claims dataframe
    input_data = input_data.merge(rx_birthdates, how = 'left', left_on = 'Member Life ID', right_on = 'Member Life ID')
    input_data = input_data.drop_duplicates()

    input_data = input_data.reset_index()
    input_data = input_data.drop('index', axis = 1)

    return input_data

def create_final_dataset(all_data, with_ages = False):
    # Performs final modifications to the input_data dataframe before the dataframe is used to train predictive models. Michael used this code to investigate his Logistic Regression Method.
    # Also calculates ages if the with_ages boolean is set to True.

    # Remove member life IDs that have conflicting Biological Genders recorded.
    # Get the Member Life IDs with two or more rows
    gender_df = all_data[['Member Life ID', 'Biological Gender']]

    gender_df = gender_df.loc[gender_df['Member Life ID'].duplicated(keep=False)].drop_duplicates()

    ids_with_two_genders = gender_df[gender_df['Member Life ID'].duplicated(keep=False)]['Member Life ID'].drop_duplicates().to_list()

    all_data = all_data[~all_data['Member Life ID'].isin(ids_with_two_genders)]
    all_data = all_data.reset_index(drop=True)

    if not with_ages:
        # Create two versions of the all_data dataframe (one with ages calculated and rows without ages removed and one with no ages calculated)
        # No Ages
        all_data_no_age = all_data.drop(['Line Service From Date', 'Header Service From Date', 'Birth Date'], axis=1)
        all_data_no_age = all_data_no_age.drop_duplicates()
        all_data_no_age = all_data_no_age.reset_index(drop=True)

        return all_data_no_age
    else:
        # Rows with Ages Only (Assuming that the Header Service From Date is the one to use)
        all_data_with_ages = all_data.loc[pd.notna(all_data['Birth Date'])]
        all_data_with_ages = all_data_with_ages.drop('Line Service From Date', axis = 1)
        all_data_with_ages[['Header Service From Date', 'Birth Date']] = all_data_with_ages[['Header Service From Date', 'Birth Date']].apply(pd.to_datetime)
        all_data_with_ages['Age'] = np.floor((all_data_with_ages['Header Service From Date'] - all_data_with_ages['Birth Date']).dt.days/365.25)
        all_data_with_ages = all_data_with_ages.drop(['Header Service From Date', 'Birth Date'], axis=1)
        all_data_with_ages = all_data_with_ages.drop_duplicates()
        all_data_with_ages = all_data_with_ages.reset_index(drop=True)

        # Removing Member Life IDs that have more than one age
        # Get the Member Life IDs with two or more rows
        age_df = all_data_with_ages[['Member Life ID', 'Age']]

        age_df = age_df.loc[age_df['Member Life ID'].duplicated(keep=False)].drop_duplicates()

        ids_with_two_ages = age_df[age_df['Member Life ID'].duplicated(keep=False)]['Member Life ID'].drop_duplicates().to_list()

        all_data_with_ages = all_data_with_ages[~all_data_with_ages['Member Life ID'].isin(ids_with_two_ages)]
        all_data_with_ages = all_data_with_ages.reset_index(drop=True)

        return all_data_with_ages

def gen_timebound_training_data(medical_claims, rx_claims, input_code):
    # Generate a Timebound training dataset for modeling
    medical_claims_timebound = create_time_bound_claims(medical_claims, input_code)

    one_hot_data = gen_one_hot_data_input(medical_claims_timebound, rx_claims)

    final_data_with_ages = create_final_dataset(one_hot_data, with_ages= True)

    final_data_with_ages.to_csv(input_code + '_with_ages_timebound.csv')

    return None