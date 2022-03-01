'''

#1 Machine Learning

'''

import enum
from random import Random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

def load_gene_data():

    # Link to `gene data` csv in L3 Data Scientist Associate Coding Test
    url = ''
    url ='https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(url)



def split_X_y(dataframe):
    
    X = dataframe.drop('Class', axis = 1)   # X includes all features except Class feature
    y = dataframe['Class']                  # y includes only Class feature
    return X, y



def undersample_X_y(X, y):
    under_sampler = RandomUnderSampler()        # Initialize RandomUnderSampler
    X_resampled, y_resampled = under_sampler.fit_resample(X, y) # Fit and sample X, y to under sampler
    return X_resampled, y_resampled



def split_into_train_test(X, y):
    # Train test split
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3)
    return X_train, y_train, X_test, y_test



def build_model(model, train_features, train_class):
    return model.fit(train_features, train_class)   # Fit model using train_features and train_class



def get_model_predictions(fitted_model, test_features):
    return fitted_model.predict(test_features)  # Predicts test_features using fitted model



def evaluate_model(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)  # Build confusion matrix

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, alpha = 0.3)            # Plots confusion matrix to 2D plot
    
    # Plots text to the 2D matrix whether the box is 0 classifications or 1 classifications
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x = j, y = i, s = conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicted Result', size = 14)
    plt.ylabel('Actual Result', size = 14)
    plt.title('Confusion Matrix', size = 20)

    # Print accuracy scores
    print('F1 Score: {:.3f}'.format(f1_score(y_true, y_pred)))
    print('Precision: {:.3f}'.format(precision_score(y_true, y_pred)))
    print('Recall: {:.3f}'.format(recall_score(y_true, y_pred)))

'''
#2: Edit-Distance Algorithm

Variables:
    - Position
    - Matching letters
    - Lettercase
    - S/Z Comparison

- If the values in "i" position are false, count += 1
- If it is the first position & 

'''

def get_edit_distance_score(string1, string2):
    i = 0
    count = 0

    while (i < len(string1)):
        
        # If the letters are either 's' or 'z', do not add to count
        if((string1[i] in ['s', 'S', 'z', 'Z']) and (string2[i] in ['s', 'S', 'z', 'Z'])):
            count += 0
        
        # Rule: if the letters are not the same, add 1 to count
        # .lower() used because letters are not equal if one is upper and one is lower
        elif(string1[i].lower() != string2[i].lower()):
            count += 1

        # Rule: if letters are in the first position, are the same,
        # and either upper or lower case, do not add to count
        if((i == 0) & (string1[i].upper() == string2[i].upper())):
            count += 0
            
        #Rule: if letters are capitalized when they shouldn't be, add 0.5 to count
        elif((i > 0) & ((string1[i].isupper()) or string2[i].isupper())):
            count += 0.5
        
        i += 1
    
    print('Distance Score between {} and {} = {}'.format(string1, string2, count))


'''
#3: Data Cleaning

'''

def load_patent_data():
    
    # Link to `patent data` csv in L3 Data Scientist Associate Coding Test`
    url = ''
    url ='https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(url)

def search_for_keyword(string, dataframe):
    return dataframe[dataframe.str.contains(string, regex = False)].count()

def search_for_phrase(string1, string2, dataframe):
    return dataframe[dataframe.str.contains(string1 + '.*' + string2, regex = True)].count()