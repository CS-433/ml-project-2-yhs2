# Introduction
This project aims to predict the sentiment of tweets using machine learning models, which include baseline embedding generation models such as GloVe, BoW, and TF-IDF, as well as classification models like logistic regression, random forest, neural network, and recurrent neural network. The main contribution of this work is DIMP-Loss, a weighted loss function that prioritizes relevant data points, improving model training. Using BERTweet-Large and the full dataset, our final model achieves 92.1% accuracy on AICrowd. We also provide data selection guidelines for practitioners.

## File Structure
- `BERT-based/`
  - Contains the BERT-based model for tweet sentiment prediction, including fine-tuning, evaluation scripts, and `run.py`. Refer to `BERT-based/README.md` for more details.
- `Glove/`
  - Scripts to create embeddings based on GloVe and train different models for prediction. Refer to `Glove/README.md` for more details.
- `EDA/`
  - `EDA.ipynb`: Contains basic exploratory data analysis (EDA) of the dataset.
  - `GPT2_pretrain.py`: Pretrain the GPT2 model on the full training dataset and compute perplexity on the test set to compare the distribution similarity between training and test data.
- `BoW_TFIDF/`
  - `helper.py`: Contains helper functions used across models for BoW and TF-IDF embeddings.
  - `{model_name}.ipynb`: Scripts for hyperparameter tuning on `{model_name}`, such as logistic regression, naive Bayes, and random forest, using BoW or TF-IDF embeddings.
- `Data`
  - Contains dataset, simply download twitter-datasets form AICrowd inside.

## Install Dependencies
```bash
conda env create -f environment.yml
conda activate ml
```

## Get the dataset for validation
For most of our experiment, we used 10% of the original training set, you can find it on wandb [https://wandb.ai/hsunyu/epfl_ml_project2/artifacts/dataset/twitter_dataset_1/v0]. Or running our code could automatically download the dataset.

## Hyperparameter Tuning
Below shows hyperparameter tuning done from Wandb:
- Logistic regression for BoW/TF-IDF [https://wandb.ai/cr7_reunited-cr7/epfl_ml_project2/sweeps/ebsgyfi0?nw=nwusercr7_reunited]
- Random forest for BoW/TF-IDF [https://wandb.ai/cr7_reunited-cr7/epfl_ml_project2/sweeps/al89gxt6?nw=nwusercr7_reunited]
- Naive Bayes for BoW/TF-IDF []

## Contribution Highlights
- **DIMP-Loss**: A weighted loss function that prioritizes relevant data points, improving model training efficiency.
- **Final Model Performance**: Utilizing BERTweet-Large on the full dataset, achieving 92.1% accuracy on AICrowd.
- **Practical Guidelines**: Recommendations for data selection to enhance model performance.
