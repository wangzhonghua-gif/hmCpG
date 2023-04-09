import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Normalization
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')
X_train = data1.iloc[:, 1:]
Y_train = data1.iloc[:, 0]
X_valid = data2.iloc[:, 1:]
Y_valid = data2.iloc[:, 0]
X_test = data3.iloc[:, 1:]
Y_test = data3.iloc[:, 0]


class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss', patience=50, verbose=1, start_epoch=0,
                 restore_best_weights=True):
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

def build_model(hp):
    tf.random.set_seed(123)
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=5, max_value=20, step=1), input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(15, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(13, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(16, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(10, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(5, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(12, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(16, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(10, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(5, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Normalization())
    model.add(Dropout(rate=hp.Float('rate', min_value=0, max_value=0.1, step=0.01)))
    model.add(Dense(1, activation='selu', kernel_initializer='lecun_normal'))
    optimizer = tf.keras.optimizers.SGD(hp.Float('learning_rate', min_value=0.001, max_value=0.015, step=0.001))
    model.compile(loss='mse', optimizer=optimizer)
    return model
tuner = RandomSearch(build_model, objective='val_loss', max_trials=200)
tuner.search_space_summary()
tuner.search(X_train, Y_train, epochs=1000, batch_size=1, validation_data=(X_valid, Y_valid), callbacks=[CustomStopper()])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.get('units'))
print(best_hps.get('rate'))
print(best_hps.get('learning_rate'))