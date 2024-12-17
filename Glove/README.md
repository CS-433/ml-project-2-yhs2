# GloVe based classification


### Generating Word Embeddings: 

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```

Now given the co-occurrence matrix and the vocabulary, it is not hard to train GloVe word embeddings, that is to compute an embedding vector for wach word in the vocabulary. We suggest to implement SGD updates to train the matrix factorization, as in

```glove_solution.py```

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets `pos_train_full.txt`, `neg_train_full.txt`

### Building a Text Classifier:

In ```validation.py```, we use NN_classifier(), LogisticRegression(), RandomForestClassifier(), XGBClassifier(), GaussianNB() as classifier, which takes the sentence embedding as input. The way to calculate the sentence embedding in this file contains 1. take the mean value of word embeddings in a sentence. 2. concatenate the word embeddings and then
pad/truncate it to a fixed length. You can change 'mode' in prepare_data() function to select the method.

We also use RNN for this task, which could directly take the sequence of word embeddings as input.

