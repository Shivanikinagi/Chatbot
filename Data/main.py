import random  # For shuffling data
import json  # To handle JSON files
import pickle  # To save and load objects
import numpy as np  # For numerical computations
import tensorflow as tf  # To build and train the neural network

import nltk  # Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # For word lemmatization

# Initialize the lemmatizer to normalize words
lemmatizer = WordNetLemmatizer()

# Load the intents JSON file
intents = json.loads(
    open(
        r"F:\shivani\VSCode\Chatbot\Data\intents.json"
    ).read()
)

# Initialize lists to store words, classes, and documents
words = []  # To store unique words from patterns
classes = []  # To store unique tags from intents
documents = []  # To store tokenized words with their corresponding tags
ignoreLetters = ["?", "!", ".", ","]  # Punctuation to ignore

# Loop through each intent in the JSON file
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each pattern into words
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)  # Add words to the words list
        documents.append((wordList, intent["tag"]))  # Pair words with their tag
        
        # Add the tag to classes if it's not already present
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # Sort and remove duplicates

# Sort classes alphabetically
classes = sorted(set(classes))

# Save words and classes for later use
pickle.dump(words, open("words.pkl", "wb"))  # Save words as a pickle file
pickle.dump(classes, open("classes.pkl", "wb"))  # Save classes as a pickle file

# Initialize training data and output vector
training = []  # To hold the training data
outputEmpty = [0] * len(classes)  # Output vector initialized to zeros

# Create the training data
for document in documents:
    bag = []  # Bag of Words for each pattern
    wordPatterns = document[0]  # List of words in the pattern
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Lemmatize words
    
    # Create a Bag of Words array
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    # Create output vector (one-hot encoded)
    outputRow = list(outputEmpty)  # Start with all zeros
    outputRow[classes.index(document[1])] = 1  # Set the index of the current class to 1
    
    # Add the bag of words and the output row to training data
    training.append(bag + outputRow)

# Shuffle the training data and convert it to a NumPy array
random.shuffle(training)  # Shuffle to improve training performance
training = np.array(training)  # Convert to NumPy array

# Split the data into inputs (X) and outputs (Y)
trainX = training[:, : len(words)]  # Features (Bag of Words)
trainY = training[:, len(words) :]  # Labels (One-hot encoded classes)

# Build the neural network model
model = tf.keras.Sequential()  # Initialize the Sequential model

# Input layer with 128 neurons and ReLU activation
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer to prevent overfitting

# Hidden layer with 64 neurons and ReLU activation
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))  # Another dropout layer

# Output layer with Softmax activation for multi-class classification
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

# Compile the model with Stochastic Gradient Descent optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
hist = model.fit(
    np.array(trainX),  # Input data
    np.array(trainY),  # Output labels
    epochs=400,  # Number of iterations
    batch_size=5,  # Number of samples per gradient update
    verbose=1  # Display progress
)

# Save the trained model
model.save("chatbot_model.h5", hist)  # Save model and training history
print("Done")  # Print confirmation message
