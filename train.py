import tensorflow as tf 
import logging, pandas
import numpy as np
import time
from sklearn import metrics
from matplotlib import pyplot as plt

from model import Model
from hyperparams import Hyperparams
from data_loader import train_data_loader, test_data_loader, prediction_dataframe, hist_data, untransformed_price


#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

H = Hyperparams()

train_batch_generator = train_data_loader(H.train_batch_size, H.num_train)
test_batch_generator = test_data_loader(H.test_batch_size, H.num_train)
prediction_dataframe_gen = prediction_dataframe()
scaler = prediction_dataframe_gen.get_scaler()
logger.info("Generators instantiated")

model = Model().get_model()
logger.info("Model loaded")

model.compile(optimizer='RMSProp', loss='mean_squared_error')
logger.info("Model compiled")

logger.info("Beginning training")
train_num_batch = H.num_train//H.train_batch_size
train_shuffled_batch = np.array([np.random.choice(train_num_batch, size=(train_num_batch), replace=False) for _ in range(H.num_epochs)])

test_num_batch = H.num_test//H.test_batch_size
# test_shuffled_batch = np.array([np.random.choice(test_num_batch, size=(test_num_batch), replace=False) for _ in range(H.num_epochs)])

predictions = np.zeros(shape=(test_num_batch*H.test_batch_size, 1))

train_loss = np.zeros(shape=(train_num_batch))
test_loss = np.zeros(shape=(test_num_batch))

for epoch in range(H.num_epochs):
    
    for batch_idx in train_shuffled_batch[epoch]:
        input_batch, labels_batch = train_batch_generator[batch_idx]
        train_loss[batch_idx] = model.train_on_batch(input_batch, labels_batch)
        logger.info("Epoch : {}, Step : {}, Loss : {}".format(epoch, batch_idx, train_loss[batch_idx]))
    
    model.save_weights("saved_weights/model_epoch_{}.h5".format(epoch))
    logger.info("Model weights - model_epoch_{} saved".format(epoch))
    
    avg_train_loss = np.mean(train_loss)
    
    logger.info("Avg Train Loss for Epoch : {} is {}".format(epoch, avg_train_loss))
    time.sleep(1)

    for batch_idx in range(test_num_batch):
        input_batch, labels_batch = test_batch_generator[batch_idx]
        predictions[batch_idx*H.test_batch_size:(batch_idx+1)*H.test_batch_size] = model.predict_on_batch(input_batch)
        test_loss[batch_idx] = metrics.mean_squared_error(labels_batch, predictions[batch_idx*H.test_batch_size:(batch_idx+1)*H.test_batch_size])
        logger.info("Epoch : {}, Step : {}, Loss : {}".format(epoch, batch_idx, test_loss[batch_idx]))
    
    avg_test_loss = np.mean(test_loss)
    
    logger.info("Avg Test Loss for Epoch : {} is {}".format(epoch, avg_test_loss))
    
    predictions = scaler.inverse_transform(predictions)
    predictions_dataframe = prediction_dataframe_gen.get_empty_dataframe()
    for i in range(predictions.shape[0]):
        predictions_dataframe.iloc[i] = predictions[i, 0]
    predictions_dataframe.to_csv("predictions/epoch_{}_predictions.csv".format(epoch))