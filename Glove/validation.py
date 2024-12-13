import numpy as np
import pickle
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch

def load_embeddings(embedding_file):
    return np.load(embedding_file)

def load_vocab(vocab_file):
    with open(vocab_file, "rb") as f:
        return pickle.load(f)

def get_word_vector(word, vocab, embeddings):
    index = vocab.get(word, -1)
    if index == -1:
        return np.zeros(embeddings.shape[1])  # Handle out-of-vocabulary words
    return embeddings[index]

def text_to_embedding(text, vocab, embeddings):
    words = text.strip().split()
    vectors = [get_word_vector(word, vocab, embeddings) for word in words]
    return np.mean(vectors, axis=0)  # Average the word vectors to get a single vector for the text

def prepare_data(df, vocab, embeddings):
    X, y = [], []
    for idx, line in df.iterrows():
        X.append(text_to_embedding(line['sentence1'], vocab, embeddings))
        y.append(line['label'])
    return np.array(X), np.array(y)

# simple neural network classifier
class NN_classifier:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        batch_size = 128

        for epoch in range(10):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = self.model(X)
        print(y_pred[:100])
        return [1 if y > 0.5 else 0 for y in y_pred]

if __name__ == "__main__":
    wandb.login(key='28b1af102774ea945af3ad3deeb378ef8541e375')
    run = wandb.init(name='load_tweet_dataset_1',
                    project='epfl_ml_project2', 
                    tags=['load_dataset'],
                    job_type='for_testing')
    # load the dataset from wandb
    artifact = run.use_artifact('hsunyu/epfl_ml_project2/twitter_dataset_1:v0')
    artifact_dir = artifact.download()

    train_df = pd.read_json(artifact_dir + '/train.json', lines=True)
    val_df = pd.read_json(artifact_dir + '/val.json', lines=True)
    test_df = pd.read_json(artifact_dir + '/test.json', lines=True)

    embeddings = load_embeddings("validation/embeddings.npy")
    vocab = load_vocab("vocab.pkl")

    # Prepare data
    X_train, y_train = prepare_data(train_df, vocab, embeddings)
    X_val, y_val = prepare_data(val_df, vocab, embeddings)
    X_test, y_test = prepare_data(test_df, vocab, embeddings)

    # Standardize the data
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Train on different classifiers, logistic regression, random forest, xgboost
    classifiers = [NN_classifier(), LogisticRegression(), RandomForestClassifier(), XGBClassifier()]

    for classifier in classifiers:
        if classifier != RandomForestClassifier():
            y_train = [1 if y == 1 else 0 for y in y_train]
            y_val = [1 if y == 1 else 0 for y in y_val]
            y_test = [1 if y == 1 else 0 for y in y_test]
        else:
            y_train = [1 if y == 1 else -1 for y in y_train]
            y_val = [1 if y == 1 else -1 for y in y_val]
            y_test = [1 if y == 1 else -1 for y in y_test]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        print(y_pred[:100])
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        print("Classifier: ", classifier)
        print(f"Accuracy: {accuracy}, F1: {f1}")
        wandb.log({"Classifier": str(classifier), "Accuracy": accuracy, "F1": f1})



