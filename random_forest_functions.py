# Loading in required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score, plot_confusion_matrix, confusion_matrix, classification_report, accuracy_score, precision_recall_curve
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, cross_val_score
import seaborn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

# Creating a function to automatically identify possible predictor ICD10 codes. Use this function in other notebooks.
def get_predictor_codes(edges, exclusions, input_code, limit = 20):
    filtered_edges = edges.loc[edges['Edge'].str.contains(input_code)].sort_values(by = ['Weight', 'Source', 'Target'], ascending = [False, True, True]).reset_index(drop=True)

    # If there are exclusions remove edges where the other base code is in the exclusion list 
    if len(exclusions) != 0:
        filtered_edges = filtered_edges.loc[~filtered_edges['Edge'].str.contains('|'.join(exclusions))].reset_index(drop=True)

    # Return the other base code of the pair
    other_codes = [''] * limit
    other_desc = [''] * limit
    weights = [''] * limit

    for i in range(limit):
        current_source = filtered_edges['Source'].iloc[i]
        current_source_desc = filtered_edges['Source Description'].iloc[i]
        current_target = filtered_edges['Target'].iloc[i]
        current_target_desc = filtered_edges['Target Description'].iloc[i]

        if current_source == input_code:
            other_codes[i] = current_target
            other_desc[i] = current_target_desc
        else:
            other_codes[i] = current_source
            other_desc[i] = current_source_desc
        weights[i] = filtered_edges['Weight'].iloc[i]
    return (other_codes, other_desc, weights)

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

def gen_one_hot_data_input_base_codes(medical_claims, rx_claims, base_codes):
    # Takes the input medical_claims and rx_claims dataframes and outputs a dataset with one-hot encoded base claims columns and Biological Gende, Line Service From Date, Header Service From Date, and Birth Date columns.
    # Differs from the function below in that this function requires that the user input a list of base codes they want added to the one-hot encoded dataframe.
    # Inputs: 
    # medical_claims = a dataframe containing the medical claims data where the icd10 base codes have been melted into one column (datasets used include one with all the icd10 codes and one with only primary codes.)
    # rx_claims = a dataframe containing the rx claims data
    # Get the unique Patient Life IDs from the medical claims data
    unique_patient_ids = medical_claims['Member Life ID'].unique()

    # Create a new dataframe to hold input data
    input_data = pd.DataFrame({'Member Life ID':unique_patient_ids})

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

# Creating a function to generate models
def gen_rf_model(input_dataset, input_code, edges, exclusion_list, best_param, limit = 20, random_seed = None, test_size = 0.20, num_folds = 10, max_iter = 250):
    # Parameters: 
    # dataframe_no_ages = pd.dataframe with one-hot-encoded ICD10 code variables and a biological gender variable
    # dataframe_with_ages = pd.dataframe with one-hot-encoded ICD10 codes,  a biological gender variable, and an age variable 
    # input_code = ICD10 diagnosis code you want to predict
    # edges = Data frame of edges between different ICD10 codes
    # exclusion_list = list of icd10 codes ignore when selecting predictor variables
    # limit = number of predictor variables to select (default = 20)
    # include_age = determines if the function will include or exclude age from the predictor variables included (True/False)
    # random_seed = random seed to use in the train_test_split and LogisticRegressionCV functions if a number is input
    # num_folds = number folds to use in Logistic Regression Cross Validation
    # max_iter = number used in the max_iter input in the LogisticRegressionCV functions
    
    # Set F = 1 and M = 0 in the Biological Gender Column
    input_dataset = input_dataset.replace({'F':1, 'M':0})

    (predictor_codes, code_descriptions, weights) = get_predictor_codes(edges, exclusion_list, input_code)
    
    # Set includes_age to True if 'Age' is a column in the input_dataset 
    includes_age = False
    if 'Age' in input_dataset.columns:
        includes_age = True

    # Creating a list of predictor variables to use
    all_variables = [input_code] + predictor_codes + ['Biological Gender']
    
    if includes_age == True:
        all_variables = all_variables + ['Age']

    # Generating a dataset model using the predictor variables that were auto selected.
    target_dataset = input_dataset[all_variables]

    # Creating the predictor matrix (X) and class labels (y) and splitting into training and test sets
    y = target_dataset[input_code]
    X = target_dataset.drop(input_code, axis =1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size, random_state = random_seed)
    num_of_estimators = best_param['n_estimators']
    max_depth = best_param['max_depth']
    # Actual output Logiistic Regression Model
    target_model = RandomForestClassifier(random_state = random_seed, class_weight='balanced', n_estimators=num_of_estimators, max_depth=max_depth).fit(X_train,y_train)
    predicted_probabilities = target_model.predict_proba(X_test)


    # Output to let the user know which ICD10 codes were used as predictors and their Weights in the edges dataset.
    predictor_codes_df = pd.DataFrame({'Code':predictor_codes, 'Description':code_descriptions, 'Weight': weights})

    # Dictionary to provide the user with the data used to train and test the data
    test_train_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return (target_model, predicted_probabilities, predictor_codes_df, test_train_dict)


def gen_rf_model_timebound(medical_claims, rx_claims, input_code, edges, exclusion_list, best_param, limit = 20, random_seed = None, test_size = 0.20, num_folds = 10, max_iter = 250):
    medical_claims_timebound = create_time_bound_claims(medical_claims, input_code)
    (predictor_codes, code_descriptions, weights) = get_predictor_codes(edges, exclusion_list, input_code)

    base_codes = [input_code] + predictor_codes

    one_hot_data = gen_one_hot_data_input_base_codes(medical_claims_timebound, rx_claims, base_codes)

    final_data_with_ages = create_final_dataset(one_hot_data, with_ages= True)

    target_model, predicted_probabilities, predictor_codes_df, test_train_dict = gen_rf_model(final_data_with_ages, input_code, edges, exclusion_list, best_param, random_seed = random_seed)

    return (target_model, predicted_probabilities, predictor_codes_df, test_train_dict)

def hyperparameter_tuning_timebound(medical_claims, rx_claims, input_code,edges, exclusion_list,  random_seed=None, test_size=0.20,
                          num_folds=10, max_iter=250):
    medical_claims_timebound = create_time_bound_claims(medical_claims, input_code)
    (predictor_codes, code_descriptions, weights) = get_predictor_codes(edges, exclusion_list, input_code)
    base_codes = [input_code] + predictor_codes
    one_hot_data = gen_one_hot_data_input_base_codes(medical_claims_timebound, rx_claims, base_codes)

    final_data_with_ages = create_final_dataset(one_hot_data, with_ages=True)

    input_dataset = final_data_with_ages.replace({'F': 1, 'M': 0})
    best_params=hyperparameter_tuning(input_dataset, input_code, predictor_codes)
    return best_params
# Creating a function to generate models
def hyperparameter_tuning(input_dataset, input_code,predictor_codes,  random_seed=None, test_size=0.20,
                          num_folds=10, max_iter=250):
    # Parameters:
    # dataframe_no_ages = pd.dataframe with one-hot-encoded ICD10 code variables and a biological gender variable
    # dataframe_with_ages = pd.dataframe with one-hot-encoded ICD10 codes,  a biological gender variable, and an age variable
    # input_code = ICD10 diagnosis code you want to predict
    # edges = Data frame of edges between different ICD10 codes
    # exclusion_list = list of icd10 codes ignore when selecting predictor variables
    # limit = number of predictor variables to select (default = 20)
    # include_age = determines if the function will include or exclude age from the predictor variables included (True/False)
    # random_seed = random seed to use in the train_test_split and LogisticRegressionCV functions if a number is input
    # num_folds = number folds to use in Logistic Regression Cross Validation
    # max_iter = number used in the max_iter input in the LogisticRegressionCV functions

    # Set F = 1 and M = 0 in the Biological Gender Column
    input_dataset = input_dataset.replace({'F':1, 'M':0})

    # (predictor_codes, code_descriptions, weights) = get_predictor_codes(edges, exclusion_list, input_code)

    # Set includes_age to True if 'Age' is a column in the input_dataset
    includes_age = False
    if 'Age' in input_dataset.columns:
        includes_age = True

    # Creating a list of predictor variables to use
    all_variables = [input_code] + predictor_codes + ['Biological Gender']

    if includes_age == True:
        all_variables = all_variables + ['Age']

    # Generating a dataset model using the predictor variables that were auto selected.
    target_dataset = input_dataset[all_variables]

    # Creating the predictor matrix (X) and class labels (y) and splitting into training and test sets
    y = target_dataset[input_code]
    X = target_dataset.drop(input_code, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    grid_params = {
        'n_estimators': range(100, 500, 100),
        'max_depth': range(2, 10, 2)
    }

    random_forest_model = RandomForestClassifier()

    grid = GridSearchCV(random_forest_model, param_grid=grid_params, cv=5)
    grid.fit(X_train, y_train)

    return grid.best_params_
def get_accuracy_sensitivity_for_all_threshold(model, test_train_dict, pred_prob, input_code, input_type='with',threshold=[.3, .4, .5]):
    for ind_threshold in threshold:
        model_prediction = (pred_prob [:,1] >= ind_threshold).astype('int')
        accuracy = accuracy_score(test_train_dict['y_test'], model_prediction)
        cm=confusion_matrix(test_train_dict['y_test'], model_prediction)
        sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
        print(input_code+" with age "+input_type+" Timebound Models Accuracy is {0} and Sensitivity is {1} for threshold {2}".format(accuracy, sensitivity, ind_threshold))

def get_accuracy_sensitivity(model, test_train_dict, pred_prob, input_code, input_type='with',threshold=.5):

    model_prediction = (pred_prob [:,1] >= threshold).astype('int')
    accuracy = accuracy_score(test_train_dict['y_test'], model_prediction)
    cm=confusion_matrix(test_train_dict['y_test'], model_prediction)
    sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
    print(input_code+" with age "+input_type+" Timebound Models Accuracy is {0} and Sensitivity is {1} for threshold {2}".format(accuracy, sensitivity, threshold))
    return (accuracy, sensitivity, cm, model_prediction)

def gen_rf_plot(random_forest_model, test_train_dict,input_code, model_prediction, accuracy,sensitivity,cm , type="with age and with Timebound"):
    rfc_cv_score = cross_val_score(random_forest_model, test_train_dict['X_train'], test_train_dict['y_train'], cv=10, scoring='roc_auc')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    ax.set_xlabel(input_code)
    plt.title("Confusion Matrix for " + input_code + " " + type)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize='x-large')
    plt.show()
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(test_train_dict['y_test'], model_prediction))
    print('\n')

    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')

    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

    print("=== Sensitivity of the Model ===")
    print(sensitivity)

    ns_probs = [0 for _ in range(len(test_train_dict['y_test']))]
    fig, ax = plt.subplots(figsize=(8, 8))
    lr_probs = random_forest_model.predict_proba(test_train_dict['X_test'])
    lr_probs = lr_probs[:, 1]
    ns_auc = roc_auc_score(test_train_dict['y_test'], ns_probs)
    lr_auc = roc_auc_score(test_train_dict['y_test'], lr_probs)
    ns_fpr, ns_tpr, _ = roc_curve(test_train_dict['y_test'], ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_train_dict['y_test'], lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("AUC/ROC Curve for " + input_code + " " + type)
    # show the legend
    plt.legend()
    plt.text(.4, 0.1, "AUC Score: " + str(rfc_cv_score.mean()), fontsize='x-large')
    # show the plot
    plt.show()