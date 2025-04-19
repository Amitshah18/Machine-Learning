import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential([
    Input(shape=(2, )),        # 👈 New Input layer here
    Dense(4,input_shape=(2, ), activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=1000, verbose=0)

prediction = model.predict(x)
print("Predictions:")
print(np.round(prediction))

loss,accuracy = model.evaluate(x,y,verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')
