__author__ = 'adnan'

'''this program compares the performance of AMWR with simple regression model when predicting
a congestion point'''
 
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
from sklearn.tree import DecisionTreeRegressor
#from sklearn.utils import check_arrays


def model():

    #reading csv file into DataFrame object
    dfa = pd.read_csv('/home/adnan/Desktop/Madrid data set/2014/03-2014.csv', parse_dates=['fecha'], index_col='fecha', sep = ';')

    dfa = dfa[dfa['identif'] == '18RV21PM01']
    df_intensidad_1 = dfa[['vmed']]

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





    std_Y1 = np.asarray(Y1)
    print "standard deviation of Input is", std_Y1.std()

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



    for i in range(len(hours)):
        X.append([hours[i], mins[i]])

    I = []
    for i in range(len(hours)):
        I.append([i])

    X_predicted = []
    Y_predicted = []





    X_training = X[:1000]
    Y_training = Y[:1000]






    #initializes the model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)


    y_rbf = svr_rbf.fit(X_training, Y_training)


    TotalError = []


    X_testing = X[50:1300]
    Y_testing = Y[50:1300]
    index = I[50:1300]

    a = y_rbf.predict(X_testing)
    print a

    return index, a


#define window sizes
TrainingWindow = 20


#creating DataFrame object
df = pd.DataFrame()


#reading csv file into DataFrame object
dfa = pd.read_csv('/home/adnan/Desktop/Madrid data set/2014/03-2014.csv', parse_dates=['fecha'], index_col='fecha', sep = ';')

dfa = dfa[dfa['identif'] == '18RV21PM01']
#dfa = dfa[dfa['identif'] == 'PM10005']
#feature. intensidad or vmed
df_intensidad_1 = dfa[['vmed']]

#code for extracting data for particular time
hour = df_intensidad_1.index.hour
print len(hour)
selector_new = (( 01 <= hour) & (hour <= 23))
df_total_1 = df_intensidad_1[selector_new]



#resampling. Original data has 15 mint sampling period
df_total_1 = df_total_1.resample('5min')

#interpolation using cubic method
df_total_1 = df_total_1.interpolate(method ='cubic')


# df_total = concat([df_total_1, df_total_2, df_total_3])

df_total = df_total_1

hours = df_total.index.hour
mins = df_total.index.minute
#df_total = concat([df_total_1])


Y1= df_total.values

std_Y1 = np.asarray(Y1)
print "standard deviation of Input is", std_Y1.std()

#regression method in the library needs array in specific format.
#Converting input and output array in the required format
X=[]
Y=[]
Y11 = []

for i in range(len(df_total)):
    Y11.append(abs(Y1[i][0]))




#getting rid of 0`s
# for i in range(len(Y11)):
#     if Y11[i] > 1 and Y11[i] < 121:
#         a=Y11[i]
#         Y.append(a)



for i in range(len(Y11)):
    a=Y11[i]+1
    Y.append(a)


#index

for i in range(len(Y)):
    X.append([i])


features = []
for i in range(len(hours)):
    features.append([hours[i], mins[i]])


print len(X), len(Y)

#initializes the model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

X_predicted = []
Y_predicted = []


















#setting total length to iterate. as i is iterated through total length and
#when we use (i+24) it should be in data
total = len(X)-TrainingWindow -2

#iterations of i for every n, where n is window size. 3 in this case



PredictionWindowArray =[]
MAPE = []
TrueError = []
PredictionWindow = 2
i=0

#for s in range (0,total,1):
while(i <= total):



#i+24- 24 is window size.it takes 24 samples window to train the model
    X_training = X[i:i+TrainingWindow]
    Y_training = Y[i:i+TrainingWindow]
    X_testing = X[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
    y_testing = Y[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
    #Y_testing = Y[i+24:i+26]


    i = i + PredictionWindow

    #training and prediction

    y_rbf = svr_rbf.fit(X_training, Y_training).predict(X_testing)

    #a=y_rbf[0]
    #for j in y_rbf[0]:
    #Y_predicted.append(a)
    for j in y_rbf:
        Y_predicted.append(j)


    for k in X_testing:
        X_predicted.append(k)






    #calculating error (y_true - y_pred)/y_true
    TempError = []

    for m in range(len(y_testing)):
        a = abs((y_testing[m] - y_rbf[m])/y_testing[m])
        if a < 1:
            c = a
        b= ((y_testing[m] - y_rbf[m])/y_testing[m])
        if abs(b) < 1:
            d = b
        TempError.append(c)
        TrueError.append(d)

#converting in an array
    TempError1 = np.asarray(TempError)

#taking mean
    TempMAPE = (TempError1.mean())*100
    MAPE.append((TempMAPE))
    #it is predicting every value in a loop. all predicted values are appended in
    #an array for overall error and graph


    #if error is greater then 20%, decrease the prediction window size
    if TempMAPE > 10:
        PredictionWindow = PredictionWindow -1

        if PredictionWindow == 0:
            PredictionWindow = 1
        PredictionWindowArray.append(PredictionWindow)
    #if error is less then 5%,increase the prediction window
    if TempMAPE < 5:
        PredictionWindow = PredictionWindow +1

        if PredictionWindow > 3:
            PredictionWindow = PredictionWindow -1

        # if PredictionWindow == 6:
        #     PredictionWindow == 4
        PredictionWindowArray.append(PredictionWindow)
    else:
        PredictionWindowArray.append(PredictionWindow)



MAPEArray = np.asarray(MAPE)


print 'error is', MAPEArray.mean()

#Extracting data for testing and plotting
Y_testing = Y[TrainingWindow:]
X_testing = X[TrainingWindow:]


model_x, model_y = model()


#plot original/actual data
plt.scatter(X_testing, Y_testing, c='k', label='Actual data')

#plot predicted data
plt.plot(X_predicted, Y_predicted, c='b', label = 'AMWR based SVR')


plt.plot(model_x, model_y, c='k', label = 'Conventional SVR')

#plt.scatter(X_predicted, Y_predicted, c='red', label = 'Predicted data')

print 'length of testing and prediction array is', len(Y_testing), len(Y_predicted)

x_step =[]
for i in range(len(PredictionWindowArray)):
    x_step.append(i)


plt.xlim(138,200)
plt.ylim(-50,150)



plt.xlabel('Time (mins)')
plt.ylabel('Average Traffic Speed (km/h)')
plt.legend()


plt.figure(2)
plt.plot(PredictionWindowArray)


plt.xlim(100,200)


#converting in an array
TrueError1 = np.asarray(TrueError)



data = TrueError1

#fitting a normal distribution
#mu, std = norm.fit(data, floc = 0)

df, mu, std = t.fit(data, floc = 0)

mu1, std1 = norm.fit(data, floc=0)
print ' T dist parameteres are', a, mu, std
print 'Gaussian parameters are', mu1, std1

# Fit a exponential distribution to the data:
#mu, std = expon.fit(data, floc = 0)

plt.figure(3)
# Plot the histogram.
plt.hist(data, bins=25, normed=True, alpha=0.5)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p1 = norm.pdf(x, loc = mu1, scale = std1)
p = t.pdf(x, df, loc = mu, scale = std)
#Data_Length = len(data)
#Normal_Data = np.random.normal(mu, std, Data_Length)

#print kstest(data, 'norm')

#if to draw a exponential
#p = expon.pdf(x, loc = mu, scale = std)

kde_data = data[:, np.newaxis]
X_plot = np.linspace(xmin, xmax, 1000)[:, np.newaxis]

# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.035).fit(kde_data)
log_dens = kde.score_samples(X_plot)
#plt.plot(X_plot[:, 0], np.exp(log_dens), linewidth = 2)

plt.plot(x, p, 'k', linewidth=2, label = 't-distribution')
plt.plot(x, p1, 'r', linewidth=2, label = 'Gaussian distribution')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#plt.title(title)
plt.xlabel('Normalized Error', fontsize = 16)
plt.ylabel('Probability of Error', fontsize = 16)
#plt.ylim(0,6)
#plt.title('Sensor 1', fontsize = 17)
plt.legend()


plt.show()





