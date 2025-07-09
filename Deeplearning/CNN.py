import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# XOR Dataset
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([[0],[1],[1],[0]])

# Build DNN model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))     
model.add(Dense(1, activation='sigmoid'))             

# Compile model
model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=500, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Make predictions
predictions = model.predict(X)
print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {pred[0]:.4f} | Rounded: {round(pred[0])}")
