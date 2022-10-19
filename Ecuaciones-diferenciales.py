import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np 

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

        with tf.GradientTape() as tape:
            # Compute the loss value
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_prend = self(x, training=True)
            dy = tape2.gradient(y_prend, x)
            x_o = tf.zeros((batch_size,1))
            y_o = self(x_o, training=True)
            eq = dy + 2.*x*y_prend
            ic = y_o - 1.
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0.,ic)

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

x=tf.linspace(-2,2,100)
history = model.fit(x,epochs=500,verbose=1)
x_testv =tf.linspace(-2,2,100)
a=model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv,np.esp(-x*x))
plt.show()
exit()

