__author__ = 'adnan'
'''this program compares the performance of AMWR with other state-of-the-art 
prediction methods in terms of Mean Absolute Percentage Error'''

import numpy as np
import csv
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from scipy.stats import norm, expon  , ks_2samp, kstest, t
from pandas import *
from sklearn.neighbors import KernelDensity
#from sklearn.utils import check_arrays
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#define window sizes
TrainingWindow = 20


#creating DataFrame object
df = pd.DataFrame()


#reading csv file into DataFrame object
dfa = pd.read_csv('/home/adnan/Desktop/Madrid data set/2014/03-2014.csv', parse_dates=['fecha'], index_col='fecha', sep = ';')

dfa = dfa[dfa['identif'] == 'PM10344']

#feature. intensidad or vmed
df_intensidad_1 = dfa[['intensidad']]

#code for extracting data for particular time
hour = df_intensidad_1.index.hour
selector_new = (( 01 <= hour) & (hour <= 23))
df_total_1 = df_intensidad_1[selector_new]



#interpolation using cubic method for missing values
df_total_1 = df_total_1.resample('5min')
df_total_1 = df_total_1.interpolate(method ='cubic')

hours = df_total_1.index.hour
mins = df_total_1.index.minute




df_total = df_total_1



Y1= df_total.values





# std_Y1 = np.asarray(Y1)
# print "standard deviation of Input is", std_Y1.std()

#regression method in the library needs array in specific format.
#Converting input and output array in the required format

X=[]
Y=[]
Y11 = []

for i in range(len(df_total)):
    Y11.append(abs(Y1[i][0]))




#getting rid of 0`s and outliers
for i in range(len(Y11)):

    a=Y11[i]+1
    Y.append(a)



# for i in range(len(Y)):
#     X.append([i])

for i in range(len(hours)):
    X.append([hours[i], mins[i]])




print len(X), len(Y)
print X, Y
X_predicted = []
Y_predicted = []





X_training = X[:3000]
Y_training = Y[:3000]






#initializing different models
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
lin_reg = linear_model.LinearRegression()
dec_reg = DecisionTreeRegressor()
rand_for= RandomForestRegressor()





y_rbf = svr_rbf.fit(X_training, Y_training)
y_lin_reg = lin_reg.fit(X_training, Y_training)
y_dec_reg = dec_reg.fit(X_training, Y_training)
y_rand_for = rand_for.fit(X_training, Y_training)

TotalError = []


TrueError = []
x = 3000+(i*200)
X_testing = X[3000:4000]
Y_testing = Y[3000:4000]

a = y_rbf.predict(X_testing)
b = y_lin_reg.predict(X_testing)
c = y_dec_reg.predict(X_testing)
d =y_rand_for.predict(X_testing)

models = [a, b, c, d]

for i in models:
    counter = 0
    TempError = []
    for m in range(len(Y_testing)):
        a = abs((Y_testing[m] - i[m])/Y_testing[m])

        if a < 1:
            counter = counter + 1
            c = a
        else:
            c = 0
        if c > 0:
            TempError.append(c)



    #converting in an array
    TempError1 = np.asarray(TempError)
    print counter

    #taking mean
    TempMAPE = (TempError1.mean())*100
    #print TempMAPE
    TrueError.append(TempMAPE)
    #print TrueError
TotalError.append(TrueError)

print TotalError







