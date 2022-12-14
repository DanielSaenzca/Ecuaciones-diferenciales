
#Inciso 2b
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #Estos dos renglones son para arreglar el error OMP:Error #15
from matplotlib import pyplot as plt
import numpy as np 
import math
class ODEsolver(Sequential):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval = -2, maxval=2)
        tf.GradientTape(persistent=True)
        with tf.GradientTape() as tape:
            # Compute the loss value
            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                    tape2.watch(x)
            y_prend = self(x, training=True)
            dy = tape2.gradient(y_prend, x)
            tape1.watch(x)
            d2y_dx2 = tape1.gradient(dy, x)
            x_o = tf.zeros((batch_size,1))
            y_o = self(x_o, training=True)
            x_1 = tf.ones((batch_size, 1))
            y_1 = self(x_1, training = True)
            #Esta es la ecuación 
            eq =  y_prend + d2y_dx2  
            ic =  y_o  + 0.
            ic1 = y_1 + 0.5
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0.,ic) + keras.losses.mean_squared_error(0.,ic1)
            
        #Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss)
        #Return a dict mapping metric names to current value
        return{"loss": self.loss_tracker.result()}

model = ODEsolver()
model.add(Dense(10, activation= 'tanh', input_shape= (1,)))
model.add(Dense(1, activation= 'tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=2000,verbose=str(1))
x_testv =tf.linspace(-5,5,100)
a=model.predict(x_testv)
#plt.plot(x_testv,a)
#plt.plot(x_testv,np.esp(-x*x))
plt.figure(figsize = (10,10))

plt.plot(x_testv, a)

plt.plot(x_testv, np.cos(x) -0.5*np.sin(x) , color = "green", linestyle='dashed')
plt.show()
exit()
#Juan Daniel Saenz Carrizoza
