# GloVe based classification

### File description

- `build_vocab.sh`: Script to build the vocabulary from the training tweets.
- `cut_vocab.sh`: Script to filter the vocabulary.
- `pickle_vocab.py`: Mapping each word to a unique index.
- `cooc.py`: Generate the co-occurrence matrix from the training tweets.
- `glove_solution.py`: Using GloVe to generate word embeddings base on the co-occurrence matrix.
- `cooc_glove_for_split`: Generate the word embeddings base on the part of the training set which is used for the whole experiment. 
- `validation.py`: Contains the implementation of various classifiers (NN_classifier, LogisticRegression, RandomForestClassifier, XGBClassifier, GaussianNB, RNN_classifier) for prediction.
- `generate_test.py`: Generate the predicted labels for test set for submission.


### Generate Word Embeddings: 

Build the vocabulary. 

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
```

Generate the word embedding for the whole training set.
```bash
python3 cooc.py
python3 glove_solution.py
```

Generate the word embedding for the training set used in experiment.
```bash
python3 cooc_glove_for_split.py
```

### Hyper-parameter tuning for Random Forest

In ```validtion.py```, RF_hyperparameter_tuning() function conduct grid search to find the best combination of hyperparameters.

### Train the model:

In ```validation.py```, we use NN_classifier(), LogisticRegression(), RandomForestClassifier(), GaussianNB() as classifier, which takes the sentence embedding as input. The way to calculate the sentence embedding in this file contains 1. take the mean value of word embeddings in a sentence. 2. concatenate the word embeddings and then pad/truncate it to a fixed length. You can change 'mode' in prepare_data() function to select the method.

We also use RNN for this task, which could directly take the sequence of word embeddings as input.

```bash
python3 validation.py
```
will directly run all the models and print the result.

