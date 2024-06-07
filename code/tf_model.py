import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np


def tf_model(df):
    
    df = df.fillna(0)
    features = df.iloc[:, :5]
    labels = df.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Define the model
    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)))  # First fully connected layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    
    model.save("classifier.h5")
    
    # Print the test accuracy
    print(f'\nTest accuracy: {test_acc:.2f}')
    return test_acc
    
"""
    # Make predictions on the test data
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype("int32")  # Convert probabilities to binary labels
    
    # Print the predicted labels
    print("Predicted Labels:")
    print(predicted_labels)

  """  
  
