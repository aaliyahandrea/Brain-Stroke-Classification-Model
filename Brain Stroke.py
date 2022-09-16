#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Required Libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# In[2]:


#load Brain Stroke dataset
brainstroke = pd.read_csv('brainstroke.csv')


# In[3]:


#How many rows/columns?
brainstroke.shape


# In[4]:


#Check the first few rows of data
brainstroke.head()


# In[5]:


#Check for Missing Data
brainstroke.isna().sum()


# In[6]:


# Summary Statistics - Continuous
brainstroke.describe()


# In[7]:


#Summary Statistics - Categorical
brainstroke[['gender','ever_married','work_type','Residence_type','smoking_status']].describe()


# In[8]:


#Categorical Variables - Univariate
#Gender
brainstroke['gender'].value_counts().plot.bar()


# In[9]:


#Categorical Variables - Univariate
#Ever_married
brainstroke['ever_married'].value_counts().plot.bar()


# In[10]:


#Categorical Variables - Univariate
#work_type
brainstroke['work_type'].value_counts().plot.bar()


# In[11]:


#Categorical Variables - Univariate
#Residence_type
brainstroke['Residence_type'].value_counts().plot.bar()


# In[12]:


#Categorical Variables - Univariate
#Smoking Status
brainstroke['smoking_status'].value_counts().plot.bar()


# In[13]:


#Continuous Variables - Univariate
#Age
brainstroke.hist(column='age')


# In[14]:


#Continuous Variables - Univariate
#hypertension
brainstroke.hist(column='hypertension')


# In[15]:


#Continuous Variables - Univariate
#heart_disease
brainstroke.hist(column='heart_disease')


# In[16]:


#Continuous Variables - Univariate
#Avg_glucose_level
brainstroke.hist(column='avg_glucose_level')


# In[17]:


#Continuous Variables - Univariate
#BMI
brainstroke.hist(column='bmi')


# In[18]:


#Continuous Variables - Univariate
#Stroke
brainstroke.hist(column='stroke')


# In[39]:


#Continuous Variables - Bivariate
brainstroke.plot.scatter(x = 'age', y = 'stroke');
brainstroke.plot.scatter(x = 'hypertension', y = 'stroke');
brainstroke.plot.scatter(x = 'avg_glucose_level', y = 'stroke');
brainstroke.plot.scatter(x = 'bmi', y = 'stroke');


# In[20]:


#Categorical Variables - Bivariate
#Gender - stroke
pd.crosstab(brainstroke.gender,brainstroke.stroke).plot(kind = 'bar')


# In[21]:


#Categorical Variables - Bivariate
#ever_married - stroke
pd.crosstab(brainstroke.ever_married,brainstroke.stroke).plot(kind = 'bar')


# In[22]:


#Categorical Variables - Bivariate
#work_type - stroke
pd.crosstab(brainstroke.work_type,brainstroke.stroke).plot(kind = 'bar')


# In[23]:


#Categorical Variables - Bivariate
#Residence_type - stroke
pd.crosstab(brainstroke.Residence_type,brainstroke.stroke).plot(kind = 'bar')


# In[24]:


#Categorical Variables - Bivariate
#smoking_status - stroke
pd.crosstab(brainstroke.smoking_status,brainstroke.stroke).plot(kind = 'bar')


# In[25]:


#Categorical Variables - Bivariate
#hypertension - stroke
pd.crosstab(brainstroke.hypertension,brainstroke.stroke).plot(kind = 'bar')


# In[26]:


#Dummy Variables of Categorical Variables
CatPrefix = ['gender','ever_married','work_type','Residence_type','smoking_status']
#Remove 1 dummy each for each column to prevent Dummy Trap
Strokedummies = pd.get_dummies(brainstroke[CatPrefix], prefix = CatPrefix)
DummyStroke= pd.concat([brainstroke,Strokedummies], axis = 'columns')
StrokeFinal = DummyStroke.drop(['gender','ever_married','work_type','Residence_type','smoking_status','gender_Male','ever_married_No','work_type_Private','Residence_type_Urban','smoking_status_smokes'],axis = 'columns')
StrokeFinal.columns


# In[27]:


StrokeFinal.rename(columns={'smoking_status_formerly smoked':'smoking_status_formerlysmoked','smoking_status_never smoked':'smoking_status_neversmoked','work_type_Self-employed':'work_type_Selfemployed'}, inplace=True)


# In[29]:


#Create arrays for the features and the response variable
y = StrokeFinal['stroke'].values
X = StrokeFinal[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Female', 'ever_married_Yes', 'work_type_Govt_job',
       'work_type_Selfemployed', 'work_type_children', 'Residence_type_Rural',
       'smoking_status_Unknown', 'smoking_status_formerlysmoked',
       'smoking_status_neversmoked']].values


# In[30]:


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)
#Provide a copy of training and test data sets
pd.DataFrame(X_train).to_csv("StrokeXTrain.csv")
pd.DataFrame(X_test).to_csv('StrokeXTest.csv')
pd.DataFrame(y_train).to_csv("StrokeYTrain.csv")
pd.DataFrame(y_test).to_csv("StrokeYTest.csv")


# In[31]:


# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 11)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[32]:


# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[33]:


#Create a k-NN classifier with 2 neighbors
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)


# In[35]:


y_pred = knn.predict(X_test)
# Generate the confusion matrix and classification report
confmatrix = confusion_matrix(y_test,y_pred)
print(confmatrix)


# In[36]:


TN = confmatrix[0,0]
TP = confmatrix[1,1]
FN = confmatrix[1,0]
FP = confmatrix[0,1]

# Calculate and print the accuracy
Accuracy = (TN + TP) / (TN + FN + FP + TP)
print("Accuracy", Accuracy)

# Calculate and print the sensitivity/recall
Sensitivity = TP / (TP + FN)
print("Sensitivity/Recall", Sensitivity)

# Calculate and print the specificity
Specificity = TN / (TN + FP)
print("Specificity", Specificity)

#Calculate and print the precision
Precision = TP/(TP + FP)
print("Precision",Precision)


# In[37]:


# Compute predicted probabilities: y_pred_prob
y_pred_prob = knn.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


# In[ ]:




