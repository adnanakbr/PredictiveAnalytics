# predictive_analytics

It has 4 python scripts.

#AMWR_online_madrid_data.py
1) It reads online data in real-time from the link (http://informo.munimadrid.es/informo/tmadrid/pm.xml), parse the xml files to extract sensors which have velocity and intensity data. It saves the data, extract for specific sensor and applies AMWR to predict for next three readings in real-time.
2) It keeps updating the model as it reads new data.
3) It requires pandas and scikit-learn to be installed for running.
4) In order to run, just type python filename.py from your command prompt.

