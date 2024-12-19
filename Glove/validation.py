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
from itertools import chain
from sklearn.naive_bayes import GaussianNB
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    # Convert a text to an embedding by taking the mean of the word vectors
    words = text.strip().split()
    vectors = [get_word_vector(word, vocab, embeddings) for word in words]
    return np.mean(vectors, axis=0)  

def text_to_embedding_padding(text, vocab, embeddings, max_length):
    # Convert a text to an embedding by padding the word vectors
    words = text.strip().split()
    vectors = [get_word_vector(word, vocab, embeddings) for word in words]
    vectors = vectors[:max_length] + [np.zeros(embeddings.shape[1])] * (max_length - len(vectors))
    return np.array(vectors).flatten()

def prepare_embedding_rnn(text, vocab, embeddings):
    # Prepare the embedding for the RNN
    words = text.strip().split()
    vectors = [get_word_vector(word, vocab, embeddings) for word in words]
    return np.array(vectors)

def prepare_data(df, vocab, embeddings, mode='mean'):
    # Prepare the data for the classifier, turn the text into embeddings
    X, y = [], []
    for idx, line in df.iterrows():
        if mode == 'mean':
            X.append(text_to_embedding(line['sentence1'], vocab, embeddings))
        else:
            X.append(text_to_embedding_padding(line['sentence1'], vocab, embeddings, 20))
        y.append(line['label'])
    return np.array(X), np.array(y)

class NN_classifier:
    # Simple neural network classifier
    def __init__(self, mode='mean'):
        first_layer = 20 if mode == 'mean' else 20 * 20
        self.model = torch.nn.Sequential(
            torch.nn.Linear(first_layer, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def fit(self, X, y):
        # Fit the neural network classifier
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        batch_size = 128

        for epoch in range(10):
            for i in tqdm(range(0, len(X), batch_size)):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        # Predict the labels
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = self.model(X)
        return [1 if y > 0.5 else 0 for y in y_pred]

class RNN_classifier:
    # RNN classifier, bidirectional LSTM
    def __init__(self):
        self.model = torch.nn.LSTM(20, 50, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = torch.nn.Linear(100, 1) 
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(chain(self.model.parameters(), self.fc.parameters()), lr=0.001, weight_decay=1e-5)

    def fit(self, X, y, x_val=None, y_val=None):
        # Fit the RNN classifier
        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        lengths = torch.tensor([len(x) for x in X], dtype=torch.int64)
        batch_size = 64
        total_loss = 0
        total_accuracy = 0

        for epoch in range(30):
            for i in tqdm(range(0, len(X), batch_size)):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                lengths_batch = lengths[i:i+batch_size]

                X_padded = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True)
                X_packed = pack_padded_sequence(X_padded, lengths_batch, batch_first=True, enforce_sorted=False)
                X_packed = X_packed
                y_batch = y_batch
                _, (h_n, _) = self.model(X_packed)
                h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Concatenate the final states of the bidirectional LSTM
                y_pred = self.fc(h_n)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()
                total_loss += loss.item()
                total_accuracy += accuracy_score(y_batch, [1 if y > 0 else 0 for y in y_pred])
            total_loss /= len(X) / batch_size
            total_accuracy /= len(X) / batch_size
            print(f"Epoch {epoch}, Loss: {total_loss}, Accuracy: {total_accuracy}")

            if epoch % 5 == 0:
                if x_val is not None and y_val is not None:
                    y_val_pred = self.predict(x_val)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_f1 = f1_score(y_val, y_val_pred)
                    print(f"Validation accuracy: {val_accuracy}, F1: {val_f1}")

    
    def predict(self, X):
        # Predict the labels
        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        lengths = torch.tensor([len(x) for x in X], dtype=torch.int64)
        X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X_packed = pack_padded_sequence(X_padded, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.model(X_packed)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  
        y_pred = self.fc(h_n)
        return [1 if y > 0 else 0 for y in y_pred]

def RF_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # Hyperparameter tuning for the random forest classifier
    best_accuracy = 0
    best_params = None
    for n_estimators in [50, 100, 200]:
        for max_depth in [10, 20, 50]:
            for min_samples_split in [2, 5, 10]:
                for min_samples_leaf in [1, 2, 4]:
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (n_estimators, max_depth, min_samples_split, min_samples_leaf)
                    print(f"Accuracy: {accuracy}, n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}")
    return best_params

if __name__ == "__main__":
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

    mode = 'padding'
    # Prepare data
    X_train, y_train = prepare_data(train_df, vocab, embeddings, mode=mode)
    X_val, y_val = prepare_data(val_df, vocab, embeddings, mode=mode)
    X_test, y_test = prepare_data(test_df, vocab, embeddings, mode=mode)

    # Standardize the data
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    '''
    hyperparameters = RF_hyperparameter_tuning(X_train, y_train, X_val, y_val)
    print("Best hyperparameters: ", hyperparameters)
    '''

    # Train on different classifiers, logistic regression, random forest, xgboost, naive bayes
    classifiers = [NN_classifier(mode), 
                   LogisticRegression(), 
                   RandomForestClassifier(n_estimators=200, max_depth=100, min_samples_split=5, min_samples_leaf=2), 
                   XGBClassifier(), 
                   GaussianNB()]

    for classifier in classifiers:
        y_train = [1 if y == 1 else 0 for y in y_train]
        y_val = [1 if y == 1 else 0 for y in y_val]
        y_test = [1 if y == 1 else 0 for y in y_test]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Classifier: ", classifier)
        print(f"Accuracy: {accuracy}, F1: {f1}")
        with open("validation/results_padding.txt", "a") as f:
            f.write(f"Classifier: {str(classifier)}\n")
            f.write(f"Accuracy: {accuracy}, F1: {f1}\n")
        wandb.log({"Classifier": str(classifier), "Accuracy": accuracy, "F1": f1})

    # RNN
    rnn_classifier = RNN_classifier()
    X_train_rnn = [prepare_embedding_rnn(text, vocab, embeddings) for text in train_df['sentence1']]
    X_val_rnn = [prepare_embedding_rnn(text, vocab, embeddings) for text in val_df['sentence1']]
    X_test_rnn = [prepare_embedding_rnn(text, vocab, embeddings) for text in test_df['sentence1']]

    y_train = [1 if y == 1 else 0 for y in y_train]
    y_val = [1 if y == 1 else 0 for y in y_val]
    y_test = [1 if y == 1 else 0 for y in y_test]

    mean, std = np.mean(list(chain(*X_train_rnn)), axis=0), np.std(list(chain(*X_train_rnn)), axis=0)
    X_train_rnn = [(X - mean) / std for X in X_train_rnn]
    X_val_rnn = [(X - mean) / std for X in X_val_rnn]
    X_test_rnn = [(X - mean) / std for X in X_test_rnn]

    rnn_classifier.fit(X_train_rnn, y_train, X_val_rnn, y_val)
    y_pred = rnn_classifier.predict(X_test_rnn)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Classifier: RNN")
    print(f"Accuracy: {accuracy}, F1: {f1}")
    wandb.log({"Classifier": "RNN", "Accuracy": accuracy, "F1": f1})
    



