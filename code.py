from keras.layers import Embedding, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import keras
import scipy.stats as stats
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
import numpy
import pyswarms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import keras

# Optimized model
# The optimal iteration model times obtained in the PSO optimization process were substituted back into the CNN model
np.random.seed(10)
modelfile = 'D:\Machine\model.h5' 
def getData():
    df = pd.read_excel("D:\Machine\Dmax3.xlsx")
    x = df[['Tg','Tx','Tl']]
    y = df[['Dmax']]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    samplein = x_train.T
    sampleout = y_train.T
    return x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler

x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = getData()

model = Sequential()  # CNN
model.add(Conv1D(16,3, activation='relu', input_shape=(x_train.shape[1], 1),
                 padding="same"))
model.add(Conv1D(16,3, activation='relu', input_shape=(x_train.shape[1], 1),
                 padding="same"))
model.add(Conv1D(16,3, activation='relu', input_shape=(x_train.shape[1], 1),
                 padding="same"))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())  # Flattening layer
model.add(Dense(36))  # Output layer
model.add(Dense(1))  # Output layer
optimizers = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse',optimizer=optimizers)
model.summary()
historydata = model.fit(x_train, y_train, epochs = 898, batch_size = 6,verbose=2,validation_split=0.2) # The model was trained 898 times
model.evaluate(x_test,y_test )
model.save(modelfile,overwrite=True) 

data = model.predict(x_test)
ypre = y_scaler.inverse_transform(data)
ytest = y_scaler.inverse_transform(y_test)

ytpre = model.predict(x_train)
ytpre = y_scaler.inverse_transform(ytpre)
ytrain = y_scaler.inverse_transform(y_train)

test_pre = model.predict(x_test)
print(test_pre)
print(test_pre.shape)
print(x_test.shape)    #(199, 3)
test_pre = test_pre.reshape(-1,1) 
print(test_pre.shape)
test_pearson = stats.pearsonr(y_test.reshape(-1), test_pre.reshape(-1)) 
print('pearson:', test_pearson)




