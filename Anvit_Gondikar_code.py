#!/usr/bin/env python
# coding: utf-8

# # Mobile Price prediction using Machine learning Model

# In[1]:


## Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the Data

# In[2]:


data_train=pd.read_csv(r"C:\Users\dell\Downloads\Train_Data.csv")


# In[3]:


data_train


# In[4]:


data_train.info()


# In[5]:


data_train.describe()


# In[6]:


# Reading the Classlable data
traindata_classlabels=pd.read_csv(r"C:\Users\dell\Downloads\Traindata_classlabels.csv")


# In[7]:


traindata_classlabels


# In[8]:


# adding classlable data
data_train['Price_range']=traindata_classlabels


# In[9]:


data_train


# # Explementary Data Analysis

# In[10]:


data_train.corr()


# In[11]:


fig = plt.figure(figsize=(15,12))
im1 = sns.heatmap(data_train.corr())


# In[12]:


sns.countplot(data_train['Price_range'])


# In[13]:


plt.figure(figsize=(20,8))
sns.barplot(data=data_train,ci='sd')


# In[14]:


sns.scatterplot(data=data_train)


# In[15]:


labels4g = ["4G-supported",'Not supported']
values4g = data_train['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()


# In[16]:


labels = ["3G-supported",'Not supported']
values=data_train['three_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()


# In[17]:


sns.barplot(x="Price_range", y="ram", data=data_train)


# # Feature Engineering 

# In[18]:


data_train= data_train.drop('clock_speed',axis=1)
#data_train= data_train.drop('mobile_wt',axis=1)


# In[19]:


data_train= data_train.drop('Price_range',axis=1)


# In[ ]:





# # Classifiaction without hyperperamter tunning

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(data_train,traindata_classlabels , test_size=0.3, random_state=101)


# # Creating & Training Logistic Regression Model

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[23]:


Lg_model = LogisticRegression()
Lg_model.fit(X_train,y_train)
y_train_pred = Lg_model.predict(X_train)
y_test_pred = Lg_model.predict(X_test)


# In[24]:


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# # Creating & Training KNN Model

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[26]:


KN= KNeighborsClassifier()
KN.fit(X_train,y_train)
y_train_pred = KN.predict(X_train)
y_test_pred = KN.predict(X_test)


# In[27]:


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# # Creating & Traing DecisionTree Model

# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[29]:


DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_train_pred = DT.predict(X_train)
y_test_pred = DT.predict(X_test)


# In[30]:


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# # Creating and Training RandomForest Model

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[32]:


RF = RandomForestClassifier()
RF.fit(X_train,y_train)
y_train_pred = RF.predict(X_train)
y_test_pred = RF.predict(X_test)


# In[33]:


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# # Creating and Training SVM Model

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[35]:


model = SVC() 
model.fit(X_train, y_train) 
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions))


# # Classification with hyperperamter tunning¶

# In[36]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


# # KNN

# In[37]:


param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance']
              
                 }


# In[38]:


# Create KNN classifier object
Knn = KNeighborsClassifier(metric ='euclidean')


# In[39]:


# Create GridSearchCV object
grid_search = GridSearchCV(Knn, param_grid, cv=5)


# In[40]:


# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


# In[41]:


# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_


# In[42]:


# Create KNN classifier with best hyperparameters
Knn_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])


# In[43]:


# Train the KNN classifier with best hyperparameters
Knn_best.fit(X_train, y_train)


# In[44]:


# Predict on the test set
y_pred = Knn_best.predict(X_test)


# In[45]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# In[46]:


# Print results
print("Best parameters:", best_params)
print("Best score:", best_score)
print("Accuracy:", accuracy)
print("Classification report:\n", report)


# # Random Forest Classifier

# In[47]:


RF = RandomForestClassifier()


# In[48]:


param_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [10, 5, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


# In[49]:


RFC = GridSearchCV(RF, param_grid, cv=5)


# In[50]:


RFC.fit(X_train, y_train)


# In[51]:


print("Best hyperparameters:", RFC.best_params_)


# In[52]:


prediction_RFC = RFC.predict(X_test)
print("Accuracy Score:\n",accuracy_score(y_test, prediction_RFC))
print("Confusion Matrix:\n", confusion_matrix(y_test,prediction_RFC))
print("Classification Report:\n", classification_report(y_test,prediction_RFC))


# # Logistic Regression

# In[53]:


params = {
    'penalty': ['l1', 'l2'], 
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga']
}

Lg_model= LogisticRegression()

grid_search = GridSearchCV(Lg_model, params, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# Train logistic regression model with best hyperparameters
Lr = LogisticRegression(**best_params)
Lr.fit(X_train, y_train)
prediction_Lr = Lr.predict(X_test)
print("Best hyperparameters:\n", best_params)
print("Accuracy Score:\n", accuracy_score(y_test,prediction_Lr))
print("Confusion Matrix:\n", confusion_matrix(y_test,prediction_Lr))
print("Classification Report:\n", classification_report(y_test,prediction_Lr))


# # Decision Tree 

# In[54]:


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

DT = DecisionTreeClassifier()

# Perform grid search using 5-fold cross validation
grid_search_DT = GridSearchCV(DT, param_grid, cv=5)
grid_search_DT.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search_DT.best_params_)
prediction_grid_search_DT=grid_search_DT.predict(X_test)
print("Accuracy Score:\n", accuracy_score(y_test,prediction_grid_search_DT))
print("Confusion Matrix:\n", confusion_matrix(y_test,prediction_grid_search_DT))
print("Classification Report:\n", classification_report(y_test,prediction_grid_search_DT))


# # SVM

# In[55]:


param_grid = {'C': [0.1, 1, 10, 100],   
              'gamma':['scale', 'auto'],
              'kernel': ['linear','poly','rbf','sigmoid']}  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train)
print(grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print("Accuracy Score:\n", accuracy_score(y_test,grid_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test, grid_predictions))


# # Result and Final Model Analysis 

# In[56]:


data_test=pd.read_csv(r"C:\Users\dell\Downloads\testdata.csv")


# In[57]:


data_test.head()


# In[58]:


data_test= data_test.drop('clock_speed',axis=1)
#data_test= data_test.drop('mobile_wt',axis=1)
#data_test= data_test.drop('touch_screen',axis=1)


# In[59]:


predicted_price=model.predict(data_test)


# In[60]:


predicted_price


# In[61]:


data_test['price_range']=predicted_price


# In[62]:


sns.countplot(data_test['price_range'])


# In[ ]:




