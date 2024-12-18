from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import wandb


def prepare_datasets(train_df, val_df, test_df):
    """
    Prepare Bag-of-Words and TF-IDF feature matrices for train, validation, and test datasets.
    """
    train_sentences = train_df['sentence1'].tolist()
    val_sentences = val_df['sentence1'].tolist()
    test_sentences = test_df['sentence1'].tolist()

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Initialize vectorizers
    bow = CountVectorizer()
    tfidf = TfidfVectorizer()

    # Bag-of-Words
    X_train_bow = bow.fit_transform(train_sentences)
    X_val_bow = bow.transform(val_sentences)
    X_test_bow = bow.transform(test_sentences)

    # TF-IDF
    X_train_tfidf = tfidf.fit_transform(train_sentences)
    X_val_tfidf = tfidf.transform(val_sentences)
    X_test_tfidf = tfidf.transform(test_sentences)

    return {
        'bow': (X_train_bow, X_val_bow, X_test_bow, y_train, y_val, y_test),
        'tfidf': (X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test)
    }


def get_optimal_RF_params():
    """
    Get optimal hyperparameters for Random Forest from Wandb
    """
    api = wandb.Api()
    sweep = api.sweep('cr7_reunited-cr7/epfl_ml_project2/sweeps/al89gxt6')
    # Get best run parameters
    best_run = sweep.best_run(order='val_accuracy')
    best_params = best_run.config
    return best_params


def get_features(X_train, X_val, X_test, y_train, y_val, y_test, num_features, indices):
    """
    Extracting top N features from optimal Random Forest's feature importance
    """
    top_features = indices[: num_features]
    return X_train[:, top_features], X_val[:, top_features], X_test[:, top_features], y_train, y_val, y_test


def get_best_run_by_conditions(sweep_id, conditions):
    """
    Returns optimal hyperparameter given a condition
    Params:
    sweep_id: ID of sweep that tested hyperparameters of model
    conditions: Dict['str': 'str'], a dictionary that filters runs from its defined conditions, e.g. {'dataset': 'bow'} would filter runs that used BoW dataset. A key is the condition with a value for an expected value of the condition.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    filtered_runs = []
    for run in sweep.runs:
        # Filter runs that matches the condition
        if all(run.config.get(key) == value for key, value in conditions.items()):
            filtered_runs.append(run)
    # Get optimal hyperparameters from filtered runs
    best_run = max(filtered_runs, key=lambda r: r.summary.get('val_accuracy', 0))
    best_params = best_run.config
    return best_params
