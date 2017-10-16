# predictive_analytics

It has 4 python scripts.

AMWR_online_madrid_data.py:
1) It reads online data in real-time from the link (http://informo.munimadrid.es/informo/tmadrid/pm.xml), parse the xml files to extract sensors which have velocity and intensity data. It saves the data, extract for specific sensor and applies AMWR to predict for next three readings in real-time.
2) It keeps updating the model as it reads new data.
3) It requires pandas and scikit-learn to be installed for running.
4) In order to run, just type python filename.py from your command prompt.

AMWR.py:
1) It runs the AMWR on historical data. It is used to verify the prediction error of the algorithm.


comparison_congestion.py:

1)This file is used to capture the congestion point in order to compare the performance of AMWR with simple prediction model.


sota_models.py:

1) it compares the performance of AMWR with other state-of-the-art regression models in terms of Mean Absolute Percentage Error (MAPE).
