import tensorflow as tf
import numpy as np
print("TensorFlow version:", tf.__version__)

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)


# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid, kernel_initializer=tf.initializers.Constant(0.5)))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse', 'binary_accuracy'])
model.summary()

history = model.fit(x, y, batch_size=1, epochs=500)

predictions = model.predict_on_batch(x)
print(predictions)