import pandas as pd 
import tensorflow as tf
import numpy as np
import logging
from skimage.transform import rescale	
from hyperparams import Hyperparams
from sklearn.preprocessing import MinMaxScaler

def string_to_float(money_str):
	money_str = money_str.replace(",","")

	if "K" in money_str:
		money_str = money_str.replace("K", "")
		return float(money_str) * 1000

	elif "M" in money_str:
		money_str = money_str.replace("M", "")
		return float(money_str) * 1000000

	elif "-" in money_str:
		return 0
	
	else:
		return float(money_str)

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

H = Hyperparams()

scaler = MinMaxScaler(feature_range=(0, 1))

file_name = 'nifty_it_historical_data.csv'

hist_data = pd.read_csv(file_name)
hist_data = hist_data.iloc[:H.usable_data, :]
hist_data['Date'] = pd.to_datetime(hist_data.Date)
hist_data.index = hist_data['Date']
hist_data.drop('Date', axis=1, inplace=True)
hist_data.drop('Change %', axis=1, inplace=True)
hist_data.drop('Vol.', axis=1, inplace=True)

for column in hist_data.columns:
	hist_data[column] = hist_data[column].apply(string_to_float)

hist_data = hist_data.reindex(index=hist_data.index[::-1])

print(hist_data.describe())
hist_data = hist_data[['Open', 'High', 'Low', 'Price']]
untransformed_price = hist_data['Price'].copy()
dataset = hist_data.values
price_scaler = MinMaxScaler(feature_range=(0, 1))
input_scaler = MinMaxScaler(feature_range=(0, 1))
dataset[:, -1:] = price_scaler.fit_transform(dataset[:, -1:])
dataset[:, :-1] = input_scaler.fit_transform(dataset[:, :-1])

print(dataset)
print(dataset.shape)

look_back_limit = H.look_back_limit
input_features = []
labels = []
for i in range(look_back_limit, len(dataset)):
    input_features.append(dataset[i - look_back_limit:i, :])
    labels.append(dataset[i, -1])
input_features = np.array(input_features)
labels = np.array(labels)

num_train = H.num_train
num_test = H.num_test
test_batch_size = H.test_batch_size

print(input_features.shape)
print(labels.shape)

class train_data_loader(tf.keras.utils.Sequence):
	
    def __init__(self, batch_size, num_train):
        self.batch_size = batch_size
        self.num_train = num_train
        self.input = input_features[:num_train]
        self.labels = labels[:num_train]
        logger.info("Training labels loaded of shape : {}".format(self.labels.shape))

    def __len__(self):
        return self.labels.shape[0] // self.batch_size

    def __getitem__(self, idx):
        input_batch = self.input[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return input_batch, labels_batch

class test_data_loader(tf.keras.utils.Sequence):
	
    def __init__(self, batch_size, num_train):
        self.batch_size = batch_size
        self.num_train = num_train
        self.input = input_features[num_train:]
        self.labels = labels[num_train:]
        logger.info("Test labels loaded of shape : {}".format(self.labels.shape))

    def __len__(self):
        return self.labels.shape[0] // self.batch_size

    def __getitem__(self, idx):
        input_batch = self.input[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return input_batch, labels_batch

class prediction_dataframe():
    
    def __init__(self):
        self.scaler = price_scaler
        self.prediction_dataframe = hist_data['Price'].iloc[-((num_test//test_batch_size)*test_batch_size):].copy()
        self.prediction_dataframe.iloc[:] = np.zeros(shape=((num_test//test_batch_size)*test_batch_size))

    def get_scaler(self):
        return self.scaler

    def get_empty_dataframe(self):
        return self.prediction_dataframe