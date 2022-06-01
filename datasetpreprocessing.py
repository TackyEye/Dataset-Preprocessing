#import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mpt  

#read dataset and drop columns which are not required
dataset=pd.read_csv('IoT Network Intrusion Dataset.csv')
print("Initial dataset:"+'\n')
print(dataset)
dataset.drop('Timestamp', inplace=True, axis=1)
dataset.drop('Flow_ID', inplace=True, axis=1)
dataset.drop('Src_IP', inplace=True, axis=1)
dataset.drop('Src_Port', inplace=True, axis=1)
dataset.drop('Dst_IP', inplace=True, axis=1)
dataset.drop('Dst_Port', inplace=True, axis=1)

#handle infinity values
dataset =dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]

#Handle missing values by replacing them with mean value of the column
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer =imputer.fit(dataset.iloc[:,1:76])
dataset.iloc[:,1:76]= imputer.transform(dataset.iloc[:,1:76])

#Encode Categorical data
Category=LabelEncoder()
dataset.iloc[:,79]=Category.fit_transform(dataset.iloc[:,79])
dataset.iloc[:,78]=Category.fit_transform(dataset.iloc[:,78])
dataset.iloc[:,77]=Category.fit_transform(dataset.iloc[:,77])

#split the dataset into training and test set
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

#feature scaling
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print('\n'+"Preprocessed dataset:"+'\n')
print(dataset)
print('\n'+"Predicted model:"+'\n')
print(y_pred)
print('\n'+"Test model:"+'\n')
print(y_test)