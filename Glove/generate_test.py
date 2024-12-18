import numpy as np
import pickle
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_embeddings(embedding_file):
    # Load the embeddings
    return np.load(embedding_file)

def load_vocab(vocab_file):
    # Load the vocabulary 
    with open(vocab_file, "rb") as f:
        return pickle.load(f)

def get_word_vector(word, vocab, embeddings):
    # Get the word vector for a word
    index = vocab.get(word, -1)
    if index == -1:
        return np.zeros(embeddings.shape[1]) 
    return embeddings[index]

def text_to_embedding(text, vocab, embeddings):
    # Convert a text to an embedding by averaging the embeddings of the words
    words = text.strip().split()
    vectors = [get_word_vector(word, vocab, embeddings) for word in words]
    return np.mean(vectors, axis=0)  

def prepare_training_data(pos_file, neg_file, vocab, embeddings):
    # Prepare the training data
    X, y = [], []
    with open(pos_file, "r", encoding='utf-8') as f:
        for line in f:
            X.append(text_to_embedding(line, vocab, embeddings))
            y.append(1)  # Positive label
    with open(neg_file, "r", encoding='utf-8') as f:
        for line in f:
            X.append(text_to_embedding(line, vocab, embeddings))
            y.append(0)  # Negative label
    return np.array(X), np.array(y)

def apply_embeddings_to_test_data(test_data_file, vocab, embeddings, classifier, output_file):
    # Apply the embeddings to the test data and write the predictions to a file
    with open(test_data_file, "r", encoding='utf-8') as f, open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Prediction"])
        for idx, line in tqdm(enumerate(f, start=1)):
            vector = text_to_embedding(line, vocab, embeddings)
            prediction = classifier.predict([vector])
            prediction_label = 1 if prediction == 1 else -1
            writer.writerow([idx, prediction_label])

if __name__ == "__main__":
    # Load the embeddings and vocabulary
    embeddings = load_embeddings("artifacts/embeddings.npy")
    vocab = load_vocab("vocab.pkl")
    
    # Prepare the training data
    X, y = prepare_training_data("./twitter-datasets/train_pos_full.txt", "./twitter-datasets/train_neg_full.txt", vocab, embeddings)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = classifier.predict(X_val)
    
    # Generate predictions for the test data
    apply_embeddings_to_test_data("./twitter-datasets/test_data.txt", vocab, embeddings, classifier, "predictions.csv")
