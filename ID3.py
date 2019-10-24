#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import random
from pprint import pprint
import time


# In[23]:


data=pd.read_csv("C:\\Users\\Dipankar\\Desktop\\CPTS570\\HW 2\\data.csv")


# In[24]:


data.head()


# In[25]:


#remove the nan column
data= data.drop('Unnamed: 32',1)


# In[26]:


data=data.drop("id",axis=1)


# In[27]:


drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
data=data.drop(drop_list1,axis=1)


# In[28]:


data.head()


# In[29]:


data=data.rename(columns={"diagnosis":"label"})


# In[30]:


data.info()


# In[31]:


data.head()


# In[32]:


data=data[['texture_mean','area_mean','smoothness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean','texture_se','area_se','smoothness_se','concavity_se','symmetry_se','fractal_dimension_se','smoothness_worst','concavity_worst','symmetry_worst','fractal_dimension_worst','label']]
#data= data[['radius_mean','texture_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','label']]


# In[33]:


data.head()


# In[35]:


#data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# In[34]:


train, validate, test = np.split(data.sample(frac=1), [int(.7*len(data)), int(.8*len(data))])


# In[35]:


#Helper function
data_train=train.values


# In[36]:


#checking if classes are pure
def check_purity(data_train):
    
    label_column = data_train[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


# In[37]:


#Classify
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# In[40]:


#classify_data(train[train.radius_mean>15].values)


# In[38]:


#Potential Split
def get_potential_splits(data):
    
    potential_splits = {}
    #We dont need the number of rows
    _, n_columns = data.shape                 
    for column_index in range(n_columns - 1):
        #potential_splits[column_index]=[]
        values = data_train[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values
        
        #for index in range(len(unique_values)):
           # if index !=0:
              #  current_value =unique_values[index]
             #   previous_value = unique_values[index-1]
             #   potential_split=(current_value+previous_value)/2
             #   potential_splits[column_index].append(potential_split)
    
    return potential_splits
    


# In[20]:


potential_splits= get_potential_splits(data_train)


# In[39]:


#Split data 
def split_data(data,split_column,split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
        
    return data_below,data_above


# In[40]:


#data_below


# In[68]:


#split_column =0
#split_value=17



# In[41]:


#data_below,data_above=split_data(data_train,split_column,split_value)


# In[42]:


#Lowest Overall Entropy
def calculate_entropy(data):
    label_column=data[:,-1]
    _,counts=np.unique(label_column, return_counts=True)


    probabilities=counts/counts.sum()
    entropy=sum(probabilities*-np.log2(probabilities))
    
    return entropy


# In[ ]:


#calculate_entropy(data_above)


# In[43]:


def calculate_overall_entropy(data_below,data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    
    return overall_entropy


# In[ ]:


#calculate_overall_entropy(data_below,data_above)


# In[44]:


def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# In[ ]:


#potential_splits=get_potential_splits(data_train)


# In[ ]:


#determine_best_split(data_train,potential_splits)


# In[ ]:


#Decision Tree Algorithm   missing feature type https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/03.%20code%20update.ipynb


# In[45]:


def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# In[46]:


#Algotithm

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


# In[47]:


tree=decision_tree_algorithm(train)
tree


# In[48]:


example = test.iloc[0]
example


# In[49]:


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


# In[50]:


classify_example(example, tree)


# In[51]:


def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy


# In[55]:


accuracy_val = calculate_accuracy(validate, tree)
accuracy_val


# In[56]:


accuracy_test = calculate_accuracy(test, tree)
accuracy_test

