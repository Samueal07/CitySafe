#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics


# In[2]:


dataset = pd.read_excel("Dataset/crp.xlsx", sheet_name="Sheet1")


# In[3]:


dataset


# In[4]:


dataset.info()


# *  In this study, the data has been collected from the **National Crime Records Bureau (NCRB)**, India.
# *  The data provides statistics on the number of crimes commited in **19 metropolitan cities** during the year **2014 to 2021**.
# *  The 19 metropolitan cities are:<br>
#     `*  Ahmedabad        *  Bengaluru        *  Chennai   `<br>
#     `*  Coimbatore       *  Delhi            *  Ghaziabad `<br>
#     `*  Hyderabad        *  Indore           *  Jaipur    `<br>
#     `*  Kanpur           *  Kochi            *  Kolkata   `<br>
#     `*  Kozhikode        *  Lucknow          *  Mumbai    `<br>
#     `*  Nagpur           *  Patna            *  Pune      `<br>
#     `*  Surat                                             `<br>
# *  It contains the records of the 10 different category of crimes commited namely:<br>
#     `*  Murder                               *  Kidnapping                   `<br>
#     `*  Crime against women                  *  Crime against children       `<br>
#     `*  Crime Committed by Juveniles         *  Crime against Senior Citizen `<br>
#     `*  Crime against SC                     *  Crime against ST             `<br>
#     `*  Economic Offences                    *  Cyber Crimes                 `<br>

# 

# In[5]:


fig, ax = plt.subplots(11, 1, figsize=(10, 50))

for i in range(0, 11):
    ax[i].barh(dataset['City'], dataset[dataset.columns[i+2]], 0.6, color='Salmon')
    ax[i].set_title('City vs ' + dataset.columns[i+2])
plt.show()


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           margin-top:20px;
#           background-color:MistyRose"> 
#     NEW DATASET CREATION
# </p>

# In[6]:


new_df = pd.DataFrame(columns=['Year', 'City', 'Population (in Lakhs) (2011)+', 'Number Of Cases', 'Type'])
for i in range(3, 13):
    temp = dataset[['Year', 'City', 'Population (in Lakhs) (2011)+']].copy()
    temp['Number Of Cases'] = dataset[[dataset.columns[i]]]
    temp['Type'] = dataset.columns[i]
    
    new_df = pd.concat([new_df, temp])


# In[7]:


new_df


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           margin-top:20px;
#           background-color:MistyRose"> 
#     DATA PRE-PROCESSING
# </p>

# <p style="font: 15px Georgia; 
#           color: green;
#           font-style: oblique;
#           text-align: justify;"> 
#     The number of cases in each crime category column will be transformed into the crime rate per population(in lakhs).<br>
#     Crime Rate = Total Crime Cases / Population (in Lakhs)
# </p>

# In[8]:


new_df['Crime Rate'] = new_df['Number Of Cases'] / new_df['Population (in Lakhs) (2011)+']


# In[9]:


new_df


# <p style="font: 15px Georgia; 
#           color: green;
#           font-style: oblique;
#           text-align: justify;"> 
#     As the Number Of Cases Column is obsolete, it should be dropped
# </p>

# In[10]:


new_df = new_df.drop(['Number Of Cases'], axis=1)


# In[11]:


new_df


# In[12]:


# saving the new dataset as an excel file
new_df.to_excel("Dataset/new_dataset.xlsx", index=False, sheet_name ='Sheet1')


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           margin-top:20px;
#           padding:15px; 
#           background-color:MistyRose"> 
#     LOADING THE NEW DATASET
# </p>

# In[13]:


new_dataset = pd.read_excel("Dataset/new_dataset.xlsx", sheet_name="Sheet1")


# In[14]:


new_dataset


# In[15]:


new_dataset.info()


# In[16]:


new_dataset.describe()


# <p style="font: 15px Georgia; 
#           color: green;
#           font-style: oblique;
#           text-align: justify;"> 
#     The data is clean with no null values with column city and type as object/categorical Dtype.
# </p>

# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           margin-top:20px;
#           background-color:MistyRose"> 
#     LABEL ENCODING
# </p>

# <p style="font: 15px Georgia; 
#           color: green;
#           font-style: oblique;
#           text-align: justify;"> 
#     Since two columns in our dataset has categorical values i.e City and Type. <br>
#     Thus, to make them machine-readable, we must convert them to numeric labels.
# </p>

# In[17]:


le = LabelEncoder()


# In[18]:


new_dataset['City'] = le.fit_transform(new_dataset['City'])
mapping = dict(zip(le.classes_, range(len(le.classes_))))


# In[19]:


# Saving the mapping file for further use
file = open('Mappings/City_Mapping.txt', 'wt')
for key,val in mapping.items():
    print(str(key) + " - " + str(val) + '\n')
    file.write(str(key) + " - " + str(val) + '\n')


# In[20]:


new_dataset['Type'] = le.fit_transform(new_dataset['Type'])
mapping = dict(zip(le.classes_, range(len(le.classes_))))


# In[21]:


# Saving the mapping file for further use
file = open('Mappings/Type_Mapping.txt', 'wt')
for key,val in mapping.items():
    print(str(key) + " - " + str(val) + '\n')
    file.write(str(key) + " - " + str(val) + '\n')


# In[22]:


new_dataset


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:MistyRose"> 
#     SPLITTING OF DATASET FOR TRAINING / TESTING
# </p>

# In[23]:


x = new_dataset[new_dataset.columns[0:4]].values
x


# In[24]:


y = new_dataset['Crime Rate'].values
y


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)


# In[26]:


x_train


# 

# In[27]:


y_train


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:MistyRose"> 
#     MODEL CREATION
# </p>

# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:PaleTurquoise"> 
#     Support Vector Machine
# </p>

# In[28]:


model1 = svm.SVR()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)


# In[29]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:PaleTurquoise "> 
#     Nearest Neighbour
# </p> 

# In[30]:


model2 = KNeighborsRegressor(n_neighbors=2)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)


# In[31]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:PaleTurquoise "> 
#     Decision Tree Regressor
# </p>

# In[32]:


model3 = tree.DecisionTreeRegressor()
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)


# In[33]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:PaleTurquoise "> 
#     Random Forest Regressor
# </p>

# In[34]:


model4 = RandomForestRegressor(random_state=0)
model4.fit(x_train, y_train)
y_pred = model4.predict(x_test)


# In[35]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:PaleTurquoise "> 
#     Neural Networks MLPRegressor
# </p>

# In[36]:


model5 = MLPRegressor(random_state=0)
model5.fit(x_train, y_train)
y_pred = model5.predict(x_test)


# In[37]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))


# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:LightGreen ">
#     CONCLUSION:<br><br>
#     The Random Forest Regression model demonstrates the best accuracy in predicting test data among the five selected models.
# </p>

# <p style="font: 20px Georgia; 
#           color: black;
#           font-style: oblique;
#           text-align: justify;
#           padding:15px; 
#           background-color:MistyRose"> 
#     SAVING THE MODEL
# </p>

# In[38]:


import pickle


# In[39]:


#saving the model as .pkl file
pkl_filename = "Model/model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model4, file)


# In[40]:


#checking the saved model accuracy
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
score = pickle_model.score(x_test, y_test)
print(score)


# In[1]:
import os

# Create a directory for saving performance plots
if not os.path.exists('performance'):
    os.makedirs('performance')

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# Sample performance metrics (replace this with your actual metrics)
performance_metrics = {
    "SVM": {
        "MAE": 0.5,
        "MSE": 0.3,
        "R2": 0.8
    },
    "KNN": {
        "MAE": 0.4,
        "MSE": 0.25,
        "R2": 0.85
    },
    "Decision Tree": {
        "MAE": 0.45,
        "MSE": 0.28,
        "R2": 0.82
    },
    "Random Forest": {
        "MAE": 0.3,
        "MSE": 0.2,
        "R2": 0.88
    },
    "Neural Network": {
        "MAE": 0.35,
        "MSE": 0.22,
        "R2": 0.84
    }
}

# Prepare data for plotting
models = list(performance_metrics.keys())
mae = [performance_metrics[model]["MAE"] for model in models]
mse = [performance_metrics[model]["MSE"] for model in models]
r2 = [performance_metrics[model]["R2"] for model in models]

# Plot MAE
plt.figure(figsize=(10, 6))
plt.bar(models, mae, color='blue', alpha=0.7)
plt.title('Mean Absolute Error (MAE) of Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.ylim(0, max(mae) + 0.1)  # Adjust the y-axis limit for better scaling
plt.savefig("performance/performance_mae.png")
plt.close()

# Plot MSE
plt.figure(figsize=(10, 6))
plt.bar(models, mse, color='orange', alpha=0.7)
plt.title('Mean Squared Error (MSE) of Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.ylim(0, max(mse) + 0.1)  # Adjust the y-axis limit for better scaling
plt.savefig("performance/performance_mse.png")
plt.close()

# Plot R² Score
plt.figure(figsize=(10, 6))
plt.bar(models, r2, color='green', alpha=0.7)
plt.title('R² Score of Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # R² score ranges from 0 to 1
plt.savefig("performance/performance_r2.png")
plt.close()

# Optional: Plot all metrics on the same graph for comparison
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12, 7))
plt.bar(x - width, mae, width, label='MAE', color='blue')
plt.bar(x, mse, width, label='MSE', color='orange')
plt.bar(x + width, r2, width, label='R² Score', color='green')

plt.title('Model Performance Metrics Comparison')
plt.xlabel('Models')
plt.ylabel('Score')
plt.xticks(x, models)
plt.ylim(0, max(max(mae), max(mse), 1) + 0.1)  # Adjust y-axis limit
plt.legend()
plt.savefig("performance/performance_comparison.png")
plt.close()

