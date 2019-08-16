import os
import pandas as pd
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

train = pd.read_csv("train.csv")
y_train = np.log(train.pop("price"))
test = pd.read_csv("test.csv")
y_test = np.log(test.pop("price"))

f_ex = Pipeline([('dict vectorizer', DictVectorizer(sparse=False)),
                  ('std scaler', StandardScaler())])

f_ex = f_ex.fit(train.to_dict(orient="row"))
normed_train_data = f_ex.transform(train.to_dict(orient="row"))
normed_test_data = f_ex.transform(test.to_dict(orient="row"))

def build_model():
  model = keras.Sequential([
    layers.Dense(30, activation=tf.nn.relu, input_shape=[len(train.keys())]),
    layers.Dense(20, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 50 == 0:
      print("Epoch: {}\tMSE: {}\tMAE: {}".format(epoch, logs["mean_squared_error"], logs["mean_absolute_error"]))

EPOCHS = 500
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(normed_train_data, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[PrintDot()])

loss, mae, mse = model.evaluate(normed_test_data, y_test, verbose=0)

print("Test MSE: {}, MAE: {}, LOSS: {}".format(mse, mae, loss))
print model.summary()

if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/model.model')
pkl.dump(f_ex, open('model/model.f_ex',"w"))

