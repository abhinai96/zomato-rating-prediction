#importing the necessary library
import pandas as pd
df=pd.read_csv(r"F:\ml practise projects\zomato\zomato.csv") #read the file
#print(df.head())
df.rename(columns = {"approx_cost(for two people)" : "cost","rest_type":"Restaurant_type"}, inplace = True) #renaming the columns
df.cost = df.cost.astype(str) #changing the object type to string
df.cost = df.cost.apply(lambda x : x.replace(',','')).astype(float) #changing the string type to float
df=df[["online_order","book_table","votes","location","Restaurant_type","cuisines","cost","rate"]] #selecting the columns
#df.head()
#applying dummies converting the categorical to numerical
df["online_order"]=pd.get_dummies(df["online_order"])
df["book_table"]=pd.get_dummies(df["book_table"])
df["location"]=pd.get_dummies(df["location"])
df["Restaurant_type"]=pd.get_dummies(df["Restaurant_type"])
df["cuisines"]=pd.get_dummies(df["cuisines"])
import numpy as np
df['rate'] = df.rate.replace('NEW', np.NaN) #replacing all the new values to nan
df['rate'] = df.rate.replace('-', np.NaN)  #replacing all the "-" values to nan
df.rate = df.rate.astype(str) #convert object to string
df.rate = df.rate.apply(lambda x : x.replace('/5','')).astype(float) #convert string to float
# Replacing the NaN values in rate feature
df['rate'] = df['rate'].fillna(df['rate'].mean()) #replacing the nan with mean
df.cost.value_counts().mean()
df['cost'] = df['cost'].fillna(df['cost'].mean())
x=df.iloc[:,:-1]
#print(x.head())
y=df.iloc[:,-1]
#importing the train_test_Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestRegressor #importing the randomforest regresor

random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(x_train, y_train)
rf_pred = random_forest_regressor.predict(x_test)
from sklearn.metrics import r2_score #checking with r2score
e=r2_score(y_test,rf_pred)
e
# For Random Forest Regressor

# open a file where you want to store the data
import pickle
file = open('zomato2.pkl', 'wb')

# dump information to that file
pickle.dump(random_forest_regressor, file)