import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the model
model = Sequential()

# Add layers
# Input layer with 2 input features and a hidden layer with 4 neurons
model.add(Dense(4, input_dim=2, activation='tanh'))
# Output layer with 1 neuron
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions
predictions = model.predict(X)
print('Predictions:')
print(predictions.round())