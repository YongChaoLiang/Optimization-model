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
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
import numpy
import pyswarms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score

'''
After 20 iterations through PSO, the iteration number of the model with optimal accuracy is selected
'''

np.random.seed(10)
modelfile = 'D:\Machine\model.h5'
def cnn(x_train, x_test, y_train, y_test, batch_size, epochs, filters, kernel_size, stride=1):
    try:
        epochs=epochs+700 # Limit the number of iterations to 700 to 1000
        model = Sequential()  # CNN
        model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(x_train.shape[1], 1),
                         padding="same"))
        model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(x_train.shape[1], 1),
                         padding="same"))
        model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(x_train.shape[1], 1),
                         padding="same"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(36))
        model.add(Dense(1))
        optimizers = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizers)
        model.summary()
        historydata = model.fit(x_train, y_train, epochs=epochs, batch_size=6, verbose=2,
                                validation_split=0.3)
        model.evaluate(x_test, y_test)
        model.save(modelfile, overwrite=True)

        predict = model.predict(x=x_test, batch_size=batch_size)



        data = model.predict(x_test)
        ypre = y_scaler.inverse_transform(data)
        ytest = y_scaler.inverse_transform(y_test)

        ytpre = model.predict(x_train)
        ytpre = y_scaler.inverse_transform(ytpre)
        ytrain = y_scaler.inverse_transform(y_train)

        test_pre = model.predict(x_test)
        test_pre = test_pre.reshape(-1, 1)
        test_pearson = stats.pearsonr(y_test.reshape(-1), test_pre.reshape(-1))  # correlation coefficient

        print(test_pearson[0])
        return test_pearson[0]

    except:
        raise



def optimizeCNN(x_train, x_test, y_train, y_test, batch_size, kernel_size, particleDimensions, stride=1):
    try:
        numberFilters = int(particleDimensions[0])  # filter
        numberEpochs = int(particleDimensions[1])  # Number of iterations

        # Call the CNN function
        r2_tmp = cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                     batch_size=batch_size,
                     epochs=numberEpochs, filters=numberFilters, kernel_size=kernel_size, stride=stride)

        return r2_tmp

    except:
        raise



def particleIteration(particles, x_train, x_test, y_train, y_test, batch_size, kernel_size, stride=1):
    try:

        numberParticles = particles.shape[0]  # Number of particles
        allLosses = [optimizeCNN(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                 kernel_size=kernel_size, particleDimensions=particles[i], stride=stride) for i in
                     range(numberParticles)]  # Call the PSO function

        return allLosses

    except:
        raise


def callCNNOptimization(x_train, x_test, y_train, y_test, batch_size, kernel_size, numberParticles, iterations, bounds,
                        stride=1, **kwargs):
    try:

        # Gets the parameters of the pso
        psoType = kwargs.get(TYPE)
        options = kwargs.get(OPTIONS)

        dimensions = 2

        if psoType == GLOBAL_BEST:
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                      options=options, bounds=bounds)  # Call pso
        else:
            raise AttributeError
        cost, pos = optimizer.optimize(objective_func=particleIteration, x_train=x_train, x_test=x_test,
                                       y_train=y_train, y_test=y_test,
                                       batch_size=batch_size, kernel_size=kernel_size, stride=stride, iters=iterations)

        return cost, pos, optimizer  # Return data

    except:
        raise

if __name__ == "__main__":
    # Define the particle swarm initialization parameters
    GLOBAL_BEST = 'G'
    LOCAL_BEST = 'L'
    C1 = 'c1'
    C2 = 'c2'
    INERTIA = 'w'
    NUMBER_NEIGHBORS = 'k'
    MINKOWSKI_RULE = 'p'
    TYPE = 'type'
    OPTIONS = 'options'

    batch_size = 5
    kernel_size = (3,)
    stride = 1

    numberParticles = 5
    iterations = 4

    minBound = numpy.ones(2)
    maxBound = numpy.ones(2)
    maxBound[0] = 601
    maxBound[1] = 300
    bounds = (minBound, maxBound)

    options = {C1: 0.3, C2: 0.2, INERTIA: 0.9, NUMBER_NEIGHBORS: 4, MINKOWSKI_RULE: 2}
    kwargs = {TYPE: GLOBAL_BEST, OPTIONS: options}



    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.optimizers import adam_v2
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation


    def getData():
        df = pd.read_excel("D:\Machine\Dmax3.xlsx")
        x = df[['Tg', 'Tx', 'Tl']]
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


    X_train, X_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = getData()


    # Call PSO to optimize the CNN function
    cost, pos, optimizer = callCNNOptimization(x_train=X_train, x_test=X_test, y_train=y_train,
                                               y_test=y_test, batch_size=batch_size,
                                               kernel_size=kernel_size,
                                               numberParticles=numberParticles,
                                               iterations=iterations,
                                               bounds=bounds, stride=stride, **kwargs)










