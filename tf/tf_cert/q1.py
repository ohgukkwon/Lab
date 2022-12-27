import numpy as np
import tensorflow as tf
from tensorflow import keras


def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-7.0, -3.0, 1.0, 5.0, 9.0, 13.0], dtype=float)

    # YOUR CODE HERE
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(xs, ys, epochs=1000)

    return model
model = solution_model()

prediction = model.predict([10, 0])[0]
print(prediction)