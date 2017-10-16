__author__ = 'adnan'

'''this program reads online data from Madrid website, pre-proces it, 
and applies AMWR to predict next three readings'''


import urllib2
import xml.etree.ElementTree as ET
import urllib
import re
import io, random
import pandas as pd
import time
from datetime import datetime
from sklearn.svm import SVR
import datetime

#constant parameters
TrainingWindow = 15    #window size is found from historical data
PredictionWindow = 3   #prediction window/prediction horizon
time_sampling = 300	#Should be equal to data refreshing time



#function to read data
def data_traffic_read():
    req = urllib2.Request(url='http://informo.munimadrid.es/informo/tmadrid/pm.xml')

    f = urllib2.urlopen(req)
    xml_str = f.read()
    root = ET.fromstring(xml_str)
    list = []
    for location in root.findall('pm'):
        codigo = location.find('codigo').text
        #here if statement can be added to look for IDs with PM and publish it
        #into different topic
        flag = re.match('PM',codigo)
        if (flag):
            intensity = float(location.find('intensidad').text)
            speed = float(location.find('velocidad').text)
            occupancy = float(location.find('ocupacion').text)
            date = (datetime.datetime.utcnow())
            error = location.find('error').text

            if error == 'N':
                message = {'ID':codigo, 'TrafficIntensity':intensity, 'TrafficSpeed': speed, 'TrafficOccupancy':occupancy, 'Date_UTC': date}
                list.append(message)
    return list

#function to apply prediction based on svr with rbf kernel
def pred(df):
    X = []
    Y = []
    for index, row in df.iterrows():
        X.append([index.hour, index.minute])
        Y.append(row)



    #initializes the model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)


    y_rbf = svr_rbf.fit(X, Y.values.ravel())

    #predicting for next readings

    #first we take the time for the latest reading
    length = len(df)
    df_pred = df[length-1:length]

    for index, row in df_pred.iterrows():
        time_last = index


    print "time_last is {}".format(time_last)
    #extracting time for next 3 predictions

    X_pred = []
    for i in range(3):
        time_new = time_last + datetime.timedelta(seconds = (i+1)*time_sampling)
        X_pred.append([time_new.hour, time_new.minute])

    Y_pred = y_rbf.predict(X_pred)


    return X_pred, Y_pred

#adaptive movie window regression function
def AMWR(df):
    #extracting last readings equivalent to window size

    length = len(df)
    df1 = df[length-TrainingWindow:length]


    df_speed = df1[['Date_UTC','TrafficSpeed']]
    df_speed1 = df_speed.set_index('Date_UTC')
    X_speed, Y_speed = pred(df_speed1)





    df_intensity = df1[['Date_UTC','TrafficIntensity']]
    df_intensity1 = df_intensity.set_index('Date_UTC')
    X_intensity, Y_intensity = pred(df_intensity1)


    for i in range(3):
        print "Expected traffic speed at {}:{} is {}".format(X_speed[i][0], X_speed[i][1], int(Y_speed[i]))
        print "Expected traffic intensity at {}:{} is {}".format(X_intensity[i][0], X_intensity[i][1], int(Y_intensity[i]))


#main function
if __name__ == '__main__':

    total_list = []
    while(1):

        list = data_traffic_read()
        total_list = total_list + list
        df = pd.DataFrame(total_list)
        #print df
        df1 = df[df['ID'] == 'PM10005']
        #print len(df1)

        if len(df1) >= TrainingWindow:
            print "calling prediction algo"
            print "df1 is ", df1
            AMWR(df1)

        else:
            pass

        print "i am sleeping"
        time.sleep(time_sampling)


