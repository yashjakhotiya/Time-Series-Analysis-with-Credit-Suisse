import pandas as pd 
from pandas import Series
import sys
from matplotlib import pyplot as plt
from data_loader import untransformed_price

predictions = pd.read_csv("predictions/epoch_{}_predictions.csv".format(sys.argv[1]), names=['Date', 'Price'])
print(predictions)
predictions['Date']= pd.to_datetime(predictions.Date)                     
predictions.index = predictions['Date']
predictions.drop('Date', axis=1, inplace=True)

plt.figure(figsize=(16,8))
plt.plot(untransformed_price, label='Close Price history')
plt.plot(predictions, label='Predictions')
plt.show()
