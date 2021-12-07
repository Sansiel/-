# Basic Import
import numpy as np
import pandas as pd

# Vis.
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/StudentsPerformance.csv")
data.head()

data["mean_scores"] = (data["math score"] + data["reading score"] + data["writing score"]) / 3

data.describe().T
# data is distributed as normally but,
# math score has 0 point cause math is harder than other all the time.
# when I look the data, I can see min exam_score is 27.
# Values of mean and 50% is so close.
data.corr()
# I want to look before I do get_dummies

def histogramPlot(variable):
    variable.plot(kind = "hist", density = True, bins = 15)
    variable.plot(kind = "kde");

if __name__=='__main__':
    histogramPlot(data)

def groupbyFunc(data, feature):
    # The function that you can use to analyze the mean of the features you have given and their situation in the data.
    values = data[feature].value_counts()
    feature_analysis = data.groupby(feature).mean()
    return values,feature_analysis    
    
    
# Firstly
groupbyFunc(data, "parental level of education")
# Secondly
groupbyFunc(data, "race/ethnicity")
# Lastly
groupbyFunc(data, "gender")

# I have to drop values of outlier scores to take a better rmse value.
sns.boxplot( y = data["math score"])
plt.show()

sns.boxplot(y = data["reading score"] )
plt.show()

sns.boxplot(y = data["writing score"])
plt.show()

sns.boxplot(y = data["mean_scores"])
plt.show()

def drop_outliers(df,column_name,lower,upper):
    drop_outliers = df[column_name].between(df[column_name].quantile(lower), df[column_name].quantile(upper))
    
    print(str(df[column_name][drop_outliers].size) + "/" + str(df[column_name].size) + " data points remain.") 

    index_names = df[~drop_outliers].index
    return df.drop(index_names)


new_data = drop_outliers(data,"mean_scores",0.05,0.95) 

print("data:",data.shape)
print("new_data:", new_data.shape)

math_score = new_data["math score"]
reading_score = new_data["reading score"]
writing_score = new_data["writing score"]
mean_score = new_data["mean_scores"]
X_features = new_data.drop(["math score","reading score","writing score","mean_scores"],axis = 'columns') 
X_features_encoded = X_features.apply(lambda x: x.astype('category')) 
X_features_encoded = pd.get_dummies(X_features_encoded,drop_first= True)
X_features_encoded

target = mean_score
X_train, X_val, y_train, y_val = train_test_split(X_features_encoded, 
                                                      target, 
                                                      test_size=0.4, 
                                                      shuffle = True, 
                                                      random_state=1)

# true ---> real     predicted---> predict
def calculateModel(real, predict):
    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    print("rmse:",rmse)
    print("r2 score:",r2)

## Random Forest and Linear Model that I tried to calculate model
print("Random Forest Regressor")
print("------------")
rf = RandomForestRegressor(random_state=0).fit(X_train, y_train)
rf_pred = rf.predict(X_train)
print("Train set of RF")
calculateModel(y_train,rf_pred)

print("------------")
print("Test set of RF")
rf_pred_val= rf.predict(X_val)
calculateModel(y_val,rf_pred_val)

print("------------")


print("Linear Regression")
print("------------")
lr = LinearRegression(normalize=True).fit(X_train, y_train)
lr_pred = lr.predict(X_train)
print("Train set of LR")
calculateModel(y_train,lr_pred)

print("------------")
print("Test set of LR")
lr_pred_val= lr.predict(X_val)
calculateModel(y_val,lr_pred_val)