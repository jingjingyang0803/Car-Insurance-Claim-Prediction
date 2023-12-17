#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import necessary libraries

# For Dataset Handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE

#For model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

#For Metrics evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

#For plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
import graphviz


# In[ ]:


### Read the dataset
df = pd.read_csv("train.csv")

# adjust these settings to display all columns
pd.set_option('display.max_columns', None)
# Display the first 10 rows
df.head(10)


# In[ ]:


# Check for missing values
df.info(verbose = True) # result in none


# In[ ]:


df.duplicated().sum() # result in 0


# In[ ]:


# Checking the unique values in the categorial features
for col in df.select_dtypes("O"):    
    print(df[col].nunique())
    print(df[col].value_counts())
    print('========================================')


# In[ ]:


# Checking the statistical summary of the dataset
df.describe()


# In[ ]:


# Checking the categorical summary of the dataset
df.describe(include = "O")


# In[ ]:


### Pre-process the data

# List boolean columns
is_cols=[col for col in df.columns if col.startswith("is") and col!="is_claim"]
print(is_cols)

# Encoding all the boolean data into numerical values 
df = df.replace({ "No" : 0 , "Yes" : 1 })


# In[ ]:


df.info(verbose = True) # check the data type


# In[ ]:


#Removal policy_id column from the original DataFrame
df.drop(columns = "policy_id", inplace = True)

# List colomns for which dummy variables were created
categorical_cols = df.select_dtypes(include=['object']).columns
print(categorical_cols)

# Dummy encoding categorical variables
df= pd.get_dummies(df, columns=categorical_cols,drop_first=True)


# In[ ]:


df.info(verbose = True) # check the data type


# In[ ]:


# Checking the correlation among the numerical features
df.corr()


# In[ ]:


### Create input data (X) and output data (y)
X = df.drop(['is_claim'], axis=1)  # Features
y = df['is_claim']  # Target variable

# Checking whether the data is balanced 
y.value_counts() # result in minority_class/majority_class≈0.068


# In[ ]:


# The data is highly imbalanced, so perform SMOTE with a specific ratio 
# to avoid either oversampling excessively or a classifier predicts only the majority class
X, y = SMOTE(sampling_strategy=0.75, random_state=42).fit_resample(X, y)

# Inspect the distribution
y.value_counts()


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the numerical features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


def classification_model(MODEL, X_train, X_test, y_train, y_test, model_name=""):
    # Initialize the model
    model = MODEL
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print the results
    print(f"Model: {model_name}")
    print("Confusion Matrix:\n", confusion)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print()
    
    # Plot the confusion matrix heatmap
    sns.heatmap(confusion, annot=True)
    plt.show()
    
    # Visualize the Decision Tree
    if model_name == "DecisionTreeClassifier":
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=list(X.columns),
            class_names=list(map(str, df['is_claim'].unique())),  # Convert to strings
            filled=True,
            rounded=True
        )

        graph = graphviz.Source(dot_data)
        graph.render(filename='decision_tree', format='png')


# In[ ]:


classification_model(LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled, y_train, y_test,"LogisticRegression")

classification_model(DecisionTreeClassifier(max_depth=4, random_state=0, criterion='gini'), X_train_scaled, X_test_scaled, y_train, y_test,"DecisionTreeClassifier")

classification_model(RandomForestClassifier(), X_train_scaled, X_test_scaled, y_train, y_test,"RandomForestClassifier")


# In[ ]:




