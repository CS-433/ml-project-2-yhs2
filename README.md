# Tweet Sentiment Classification

### Introduction
This project aims to predict the sentiment of tweets using machine learning models, which contains baseline embedding generate model such as GloVe, Bow, TF-IDF, classification model such as logistic regression, random forest, neural network, recurrent neural network and so on. The main contribution of this work is DIMP-Loss, a weighted loss function that prioritizes relevant data points, improving model training. Using BERTweet-Large and the full dataset, our final model achieves 92.1% accuracy on AICrowd. We also provide data selection guidelines for practitioners.

### File description
- `BERT-based`: Contains the BERT based model for tweet sentiment prediction.
- `Glove`: Contains the scripts that create embedding based on Glove and different models for prediction. 
- `Sean`: Contains the scripts that create embedding based on BoW and TF-IDF and different models for prediction. 
- `GPT2-pretrain.py`: Pretrain the GPT2 model on the full train set, then apply the model to calculate the perplexity in test set to reflect the distribution similarity of train and test set.
- `EDA.ipynb`: Exploratory data analysis for the dataset.
- `load_data_example.ipynb`: The sample program to load the training data that the experiment use (part of full training data) from wandb.